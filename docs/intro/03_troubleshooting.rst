Troubleshooting
===================

See a list of common problems below and their solutions. If you have a suggestion for additional hints here,
please get in touch.

1. ``BlockingIOError: [Errno 35] Unable to open file (unable to lock file, errno = 35, error message = 'Resource temporarily unavailable')``
    This error is caused by the HDF5 library not being able to open the file. This can happen if the file is already
    open in another program. Try closing any other programs that might be using the file, and try again.
