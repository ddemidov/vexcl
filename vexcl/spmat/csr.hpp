#ifndef VEXCL_SPMAT_CSR_HPP
#define VEXCL_SPMAT_CSR_HPP

/*
The MIT License

Copyright (c) 2012-2013 Denis Demidov <ddemidov@ksu.ru>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/**
 * \file   vexcl/spmat/csr.hpp
 * \author Denis Demidov <ddemidov@ksu.ru>
 * \brief  OpenCL sparse matrix in CSR format.
 */

struct SpMatCSR : public sparse_matrix {
    const cl::CommandQueue &queue;
    size_t n;

    struct matrix_part {
        size_t nnz;
        cl::Buffer row;
        cl::Buffer col;
        cl::Buffer val;
    } loc, rem;

    SpMatCSR(
            const cl::CommandQueue &queue,
            const idx_t *row, const col_t *col, const val_t *val,
            size_t row_begin, size_t row_end, size_t col_begin, size_t col_end,
            std::set<col_t> ghost_cols
            )
        : queue(queue), n(row_end - row_begin)
    {
        auto is_local = [col_begin, col_end](size_t c) {
            return c >= col_begin && c < col_end;
        };

        cl::Context ctx = qctx(queue);

        if (ghost_cols.empty()) {
            loc.nnz = row[row_end] - row[row_begin];
            rem.nnz = 0;

            if (loc.nnz) {
                loc.row = cl::Buffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(idx_t) * (n + 1), const_cast<idx_t*>(row + row_begin));
                loc.col = cl::Buffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(col_t) * loc.nnz, const_cast<col_t*>(col + row[row_begin]));
                loc.val = cl::Buffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(val_t) * loc.nnz, const_cast<val_t*>(val + row[row_begin]));

                if (row_begin > 0) vector<idx_t>(queue, loc.row) -= row_begin;
            }
        } else {
            std::vector<idx_t> lrow;
            std::vector<col_t> lcol;
            std::vector<val_t> lval;

            std::vector<idx_t> rrow;
            std::vector<col_t> rcol;
            std::vector<val_t> rval;

            lrow.reserve(n + 1);
            lrow.push_back(0);

            lcol.reserve(row[row_end] - row[row_begin]);
            lval.reserve(row[row_end] - row[row_begin]);

            if (!ghost_cols.empty()) {
                rrow.reserve(n + 1);
                rrow.push_back(0);

                rcol.reserve(row[row_end] - row[row_begin]);
                rval.reserve(row[row_end] - row[row_begin]);
            }

            // Renumber columns.
            std::unordered_map<col_t, col_t> r2l(2 * ghost_cols.size());
            size_t nghost = 0;
            for(auto c = ghost_cols.begin(); c != ghost_cols.end(); ++c)
                r2l[*c] = nghost++;

            for(size_t i = row_begin; i < row_end; ++i) {
                for(idx_t j = row[i]; j < row[i + 1]; j++) {
                    if (is_local(col[j])) {
                        lcol.push_back(col[j] - col_begin);
                        lval.push_back(val[j]);
                    } else {
                        assert(r2l.count(col[j]));
                        rcol.push_back(r2l[col[j]]);
                        rval.push_back(val[j]);
                    }
                }

                lrow.push_back(lcol.size());
                rrow.push_back(rcol.size());
            }


            // Copy local part to the device.
            if (lrow.back()) {
                loc.nnz = lrow.back();

                loc.row = cl::Buffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes(lrow), lrow.data());
                loc.col = cl::Buffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes(lcol), lcol.data());
                loc.val = cl::Buffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes(lval), lval.data());
            }

            // Copy remote part to the device.
            if (!ghost_cols.empty()) {
                rem.nnz = rrow.back();

                rem.row = cl::Buffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes(rrow), rrow.data());
                rem.col = cl::Buffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes(rcol), rcol.data());
                rem.val = cl::Buffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes(rval), rval.data());
            }
        }
    }

    template <class OP>
    void mul(
            const matrix_part &part,
            const cl::Buffer &in, const cl::Buffer &out,
            val_t scale,
            const std::vector<cl::Event> &wait_for_it = std::vector<cl::Event>()
            ) const
    {
        static kernel_cache cache;

        cl::Context context = qctx(queue);
        cl::Device  device  = qdev(queue);

        auto kernel = cache.find(context());

        if (kernel == cache.end()) {
            std::ostringstream source;

            source << standard_kernel_header(device) <<
                "kernel void csr_spmv(\n"
                "    " << type_name<size_t>() << " n,\n"
                "    " << type_name<val_t>()  << " scale,\n"
                "    global const " << type_name<idx_t>() << " * row,\n"
                "    global const " << type_name<col_t>() << " * col,\n"
                "    global const " << type_name<val_t>() << " * val,\n"
                "    global const " << type_name<val_t>() << " * in,\n"
                "    global       " << type_name<val_t>() << " * out\n"
                "    )\n"
                "{\n"
                "    for (size_t i = get_global_id(0); i < n; i += get_global_size(0)) {\n"
                "        " << type_name<val_t>() << " sum = 0;\n"
                "        for(size_t j = row[i], e = row[i + 1]; j < e; ++j)\n"
                "            sum += val[j] * in[col[j]];\n"
                "        out[i] " << OP::string() << " scale * sum;\n"
                "    }\n"
                "}\n";

            auto program = build_sources(context, source.str());

            cl::Kernel krn(program, "csr_spmv");
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
        krn.setArg(pos++, part.row);
        krn.setArg(pos++, part.col);
        krn.setArg(pos++, part.val);
        krn.setArg(pos++, in);
        krn.setArg(pos++, out);

        queue.enqueueNDRangeKernel(krn, cl::NullRange, g_size, wgsize,
                wait_for_it.empty() ? NULL : &wait_for_it);
    }

    void mul_local(const cl::Buffer &in, const cl::Buffer &out, val_t scale, bool append) const {
        if (append) {
            if (loc.nnz) mul<assign::ADD>(loc, in, out, scale);
        } else {
            if (loc.nnz)
                mul<assign::SET>(loc, in, out, scale);
            else
                vector<val_t>(queue, out) = 0;
        }
    }

    void mul_remote(const cl::Buffer &in, const cl::Buffer &out, val_t scale,
            const std::vector<cl::Event> &wait_for_it) const
    {
        if (rem.nnz) mul<assign::ADD>(rem, in, out, scale, wait_for_it);
    }

    static std::string inline_preamble(int component, int position) {
        std::ostringstream s;

        s << type_name<val_t>() <<
          " csr_spmv_" << component << "_" << position << "(\n"
          "    global const " << type_name<idx_t>() << " * row,\n"
          "    global const " << type_name<col_t>() << " * col,\n"
          "    global const " << type_name<val_t>() << " * val,\n"
          "    global const " << type_name<val_t>() << " * in,\n"
          "    ulong i\n"
          "    )\n"
          "{\n"
          "    " << type_name<val_t>() << " sum = 0;\n"
          "    for(size_t j = row[i], e = row[i + 1]; j < e; ++j)\n"
          "        sum += val[j] * in[col[j]];\n"
          "    return sum;\n"
          "}\n";

        return s.str();
    }

    static std::string inline_expression(int component, int position) {
        std::ostringstream prm;
        prm << "prm_" << component << "_" << position << "_";

        std::ostringstream s;
        s << "csr_spmv_" << component << "_" << position << "("
          << prm.str() << "row, "
          << prm.str() << "col, "
          << prm.str() << "val, "
          << prm.str() << "vec, idx)";

        return s.str();
    }

    static std::string inline_parameters(int component, int position) {
        std::ostringstream prm;
        prm << "prm_" << component << "_" << position << "_";

        std::ostringstream s;
        s <<
          ",\n\tglobal const " << type_name<idx_t>() << " * " << prm.str() << "row"
          ",\n\tglobal const " << type_name<col_t>() << " * " << prm.str() << "col"
          ",\n\tglobal const " << type_name<val_t>() << " * " << prm.str() << "val"
          ",\n\tglobal const " << type_name<val_t>() << " * " << prm.str() << "vec";

        return s.str();
    }

    void setArgs(cl::Kernel &krn, uint device, uint &pos, const vector<val_t> &x) const {
        krn.setArg(pos++, loc.row);
        krn.setArg(pos++, loc.col);
        krn.setArg(pos++, loc.val);
        krn.setArg(pos++, x(device));
    }
};

#endif
