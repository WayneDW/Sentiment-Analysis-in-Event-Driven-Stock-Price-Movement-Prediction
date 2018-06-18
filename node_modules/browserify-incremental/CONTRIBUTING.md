# Contributor Guidelines

This project has automated tests which can be run with `npm test`. This will be
run in Travis CI when you make a Pull Request so please try to ensure that they
are passing when submitting one. If your PR adds new features please also add 
tests. If your PR fixes a bug, it might be helpful to add a test which would fail
if the bug was re-introduced.

The source code of this project is also checked for coding errors and code style
consistency using ESLint. You can run it with `npm run lint` to see any errors,
and you can automatically fix some coding style errors using `npm run lint-fix`.
