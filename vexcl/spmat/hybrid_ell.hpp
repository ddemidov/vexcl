#ifndef VEXCL_SPMAT_HYBRID_ELL_HPP
#define VEXCL_SPMAT_HYBRID_ELL_HPP

namespace vex {

template <typename val_t, typename col_t, typename idx_t, typename T>
struct SpMatHELL {
    static const col_t not_a_column = static_cast<col_t>(-1);

    size_t n, pitch;
    
    struct matrix_part {
        struct {
            size_t     width;
            cl::Buffer col;
            cl::Buffer val;
        } ell;

        struct {
            size_t     nnz;
            cl::Buffer row;
            cl::Buffer col;
            cl::Buffer val;
        } csr;
    } loc, rem;

    SpMatHELL(
            const cl::CommandQueue &queue,
            const idx_t *row, const col_t *col, const val_t *val,
            size_t row_begin, size_t row_end, size_t col_begin, size_t col_end,
            std::set<col_t> ghost_cols
            )
        : queue(queue), n(row_end - row_begin), pitch( alignup(n, 16U) )
    {
        /* 1. Get optimal ELL widths for local and remote parts. */
        {
            // Speed of ELL relative to CSR (e.g. 2.0 -> ELL is twice as fast):
            const double ell_vs_csr = 3.0;

            // Find maximum widths for local and remote parts:
            loc.ell.width = rem.ell.width = 0;
            for(size_t i = row_begin; i < row_end; ++i) {
                size_t wl = 0, wr = 0;
                for(idx_t j = row[i]; j < row[i + 1]; ++j) {
                    if (col[j] >= col_begin && col[j] < col_end)
                        ++wl;
                    else
                        ++wr;
                }

                loc.ell.width = std::max(loc.ell.width, wl);
                rem.ell.width = std::max(rem.ell.width, wr);
            }

            // Build histograms for width distribution.
            std::vector<size_t> loc_hist(loc.ell.width + 1, 0);
            std::vector<size_t> rem_hist(rem.ell.width + 1, 0);

            for(size_t i = row_begin; i < row_end; ++i) {
                size_t wl = 0, wr = 0;
                for(idx_t j = row[i]; j < row[i + 1]; ++j) {
                    if (col[j] >= col_begin && col[j] < col_end)
                        ++wl;
                    else
                        ++wr;
                }

                ++loc_hist[wl];
                ++rem_hist[wr];
            }

            auto optimal_width = [&](size_t max_width, const std::vector<size_t> &hist) {
                for(size_t i = 0, rows = n; i <= max_width; ++i) {
                    rows -= hist[i]; // Number of rows wider than i.
                    if (ell_vs_csr * rows < n) return i;
                }

                return max_width;
            };

            loc.ell.width = optimal_width(loc.ell.width, loc_hist);
            rem.ell.width = optimal_width(rem.ell.width, rem_hist);
        }

        /* 2. Count nonzeros in CSR parts of the matrix. */
        loc.nnz = 0, rem.nnz = 0;

        for(size_t i = row_beg; i < row_end; i++) {
            size_t wl = 0, wr = 0;
            for(idx_t j = row[i]; j < row[i + 1]; ++j) {
                if (col[j] >= col_begin && col[j] < col_end)
                    ++wl;
                else
                    ++wr;
            }

            if (wl > loc.ell.width) loc.nnz += wl - loc.ell.width;
            if (wr > rem.ell.width) rem.nnz += wr - rem.ell.width;
        }

        /* 3. Renumber columns. */
        std::unordered_map<col_t,col_t> r2l(2 * ghost_cols.size());
        size_t nghost = 0;
        for(auto c = ghost_cols.begin(); c != ghost_cols.end(); c++)
            r2l[*c] = nghost++;

        // Prepare ELL and COO formats for transfer to devices.
        std::vector<col_t> lell_col(pitch * loc.ell.width, not_a_column);
        std::vector<val_t> lell_val(pitch * loc.ell.width, 0);
        std::vector<col_t> rell_col(pitch * rem.ell.width, not_a_column);
        std::vector<val_t> rell_val(pitch * rem.ell.width, 0);

        std::vector<idx_t> lcsr_row;
        std::vector<col_t> lcsr_col;
        std::vector<val_t> lcsr_val;

        std::vector<idx_t> rcsr_row;
        std::vector<col_t> rcsr_col;
        std::vector<val_t> rcsr_val;

        lcsr_row.reserve(n + 1);
        lcsr_col.reserve(loc.nnz);
        lcsr_val.reserve(loc.nnz);

        rcsr_row.reserve(n + 1);
        rcsr_col.reserve(rem.nnz);
        rcsr_val.reserve(rem.nnz);

        lcsr_row.push_back(0);
        rcsr_row.push_back(0);

        for(size_t i = row_begin, k = 0; i < row_end; ++i, ++k) {
            size_t lcnt = 0, rcnt = 0;
            for(idx_t j = row[i]; j < row[i + 1]; ++j) {
                if (col[j] >= col_begin && col[j] < col_end) {
                    if (lcnt < loc.ell.width) {
                        lell_col[k + pitch * lcnt] = col[j] - col_begin;
                        lell_val[k + pitch * lcnt] = val[j];
                        ++lcnt;
                    } else {
                        lcsr_col.push_back(col[j] - col_begin);
                        lcsr_val.push_back(val[j]);
                    }
                } else {
                    assert(r2l.count(col[j]));
                    if (rcnt < rem.ell.width) {
                        rell_col[k + pitch * rcnt] = r2l[col[j]];
                        rell_val[k + pitch * rcnt] = val[j];
                        ++rcnt;
                    } else {
                        rcsr_col.push_back(r2l[col[j]]);
                        rcsr_val.push_back(val[j]);
                    }
                }
            }

            lcsr_row.push_back(lcsr_col.size());
            rcsr_row.push_back(rcsr_col.size());
        }

        /* Copy data to device */
        cl::Context ctx = qctx(queue);

        if (loc.ell.width) {
            loc.ell.col = cl::Buffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes(lell_col), lell_col.data());
            loc.ell.val = cl::Buffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes(lell_val), lell_col.data());
        }

        loc.csr.row = cl::Buffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes(lcsr_row), lcsr_row.data());

        if (loc.nnz) {
            loc.csr.col = cl::Buffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes(lcsr_col), lcsr_col.data());
            loc.csr.val = cl::Buffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes(lcsr_val), lcsr_val.data());
        }

        if (rem.ell.w) {
            rem.ell.col = cl::Buffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes(rell_col), rell_col.data());
            rem.ell.val = cl::Buffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes(rell_val), rell_val.data());
        }

        rem.csr.row = cl::Buffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes(rcsr_row), rcsr_row.data());

        if (rem.nnz) {
            rem.csr.col = cl::Buffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes(rcsr_col), rcsr_col.data());
            rem.csr.val = cl::Buffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes(rcsr_val), rcsr_val.data());
        }
    }

    template <class OP, typename T_IN, typename T_OUT>
    void mul(const matrix_part &part, const cl::Buffer &in, const cl::Buffer &out, val_t scale) const {
        static kernel_cache cache;

        cl::Context context = qctx(queue);
        cl::Device  device  = qdev(queue);

        auto kernel = cache.find(context());

        if (kernel == cache.end()) {
            std::ostringstream source;

            source << standard_kernel_header(device) <<
                "kernel void hybrid_ell_spmv(\n"
                "    " << type_name<size_t>() << " n,\n"
                "    " << type_name<val_t>()  << " scale,\n"
                "    " << type_name<size_t>() << " ell_w,\n"
                "    " << type_name<size_t>() << " ell_pitch,\n"
                "    global const " << type_name<col_t>() << " * ell_col,\n"
                "    global const " << type_name<val_t>() << " * ell_val,\n"
                "    global const " << type_name<idx_t>() << " * csr_row,\n"
                "    global const " << type_name<col_t>() << " * csr_col,\n"
                "    global const " << type_name<val_t>() << " * csr_val,\n"
                "    global const " << type_name<T_IN>()  << " * x,\n"
                "    global       " << type_name<T_OUT>() << " * y\n"
                "    )\n"
                "{\n"
                "    for (size_t i = get_global_id(0); i < n; i += get_global_size(0)) {\n"
                "        " << type_name<T_OUT>() << " sum = 0;\n"
                "        for(size_t j = 0; j < ell_w; ++j) {\n"
                "            " << type_name<col_t>() << " c = ell_col[i + j * ell_pitch];\n"
                "            if (c != ("<< type_name<col_t>() << ")(-1))\n"
                "                sum += ell_val[i + j * ell_pitch] * x[c];\n"
                "        }\n"
                "        for(size_t j = csr_row[i], e = csr_row[i + 1]; j < e; ++j)\n"
                "            sum += csr_val[j] * x[csr_col[j]];\n"
                "        y[row[i]] " << OP::string() << " scale * sum;\n"
                "    }\n"
                "}\n";

            auto program = build_sources(context, source.str());

            cl::Kernel krn(program, "hybrid_ell_spmv");
            size_t     wgs = kernel_workgroup_size(krn, device);

            kernel = cache.insert(std::make_pair(
                        context(), kernel_cache_entry(krn, wgs)
                        )).first;
        }

        cl::Kernel krn    = kernel->second.kernel;
        size_t     wgsize = kernel->second.wgsize;
        size_t     g_size = num_workgroups(device) * wgsize;

        unsigned pos = 0;
        krn.setArg(pos++, n);
        krn.setArg(pos++, scale);
        krn.setArg(pos++, part.ell.width);
        krn.setArg(pos++, pitch);
        if (part.ell.width) {
            krn.setArg(pos++, part.ell.col);
            krn.setArg(pos++, part.ell.val);
        } else {
            krn.setArg(pos++, static_cast<void*>(0));
            krn.setArg(pos++, static_cast<void*>(0));
        }
        krn.setArg(pos++, part.csr.row);
        if (part.csr.nnz) {
            krn.setArg(pos++, part.csr.col);
            krn.setArg(pos++, part.csr.val);
        } else {
            krn.setArg(pos++, static_cast<void*>(0));
            krn.setArg(pos++, static_cast<void*>(0));
        }
        krn.setArg(pos++, in);
        krn.setArg(pos++, out);

        queue.enqueueNDRangeKernel(krn, cl::NullRange, g_size, wgsize);
    }

    template <class OP, typename T_IN, typename T_OUT>
    void mul_local(const cl::Buffer &in, const cl::Buffer &out, val_t scale) const {
        mul<OP, T_IN, T_OUT>(local, in, out, scale);
    }

    template <class OP, typename T_IN, typename T_OUT>
    void mul_remote(const cl::Buffer &in, const cl::Buffer &out, val_t scale) const {
        mul<typename assign::finish<OP>::type, T_IN, T_OUT>(remote, in, out, scale);
    }
};

} // namespace vex

#endif
