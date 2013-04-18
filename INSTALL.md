Installation
------------

Since VexCL is header-only library, installation is straightforward: you just
need to copy vexcl folder somewhere and tell your compiler to scan it for
include files.

* [Gentoo Linux ebuild](https://github.com/ddemidov/ebuilds/blob/master/dev-util/vexcl)
* [Arch Linux PKGBUILD](https://aur.archlinux.org/packages/vexcl-git)


Dependencies
------------

VexCL depends on [Boost](http://www.boost.org). In particular, you may need to
link to Boost.chrono, Boost.date_time, Boost.filesystem, Boost.system,
Boost.thread.
