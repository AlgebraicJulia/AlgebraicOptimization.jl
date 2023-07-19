# AlgebraicOptimization.jl

[![Stable Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://AlgebraicJulia.github.io/AlgebraicOptimization.jl/stable)
[![Development Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://AlgebraicJulia.github.io/AlgebraicOptimization.jl/dev)
[![Code Coverage](https://codecov.io/gh/AlgebraicJulia/AlgebraicOptimization.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/AlgebraicJulia/AlgebraicOptimizatione.jl)
[![CI/CD](https://github.com/AlgebraicJulia/AlgebraicOptimization.jl/actions/workflows/julia_ci.yml/badge.svg)](https://github.com/AlgebraicJulia/AlgebraicOptimization.jl/actions/workflows/julia_ci.yml)

Building and solving (convex) optimization problems compositionally.

## TODOs:

### 📔 Set Up GitHub Pages (Public Repos Only)

1. Follow the Usage steps above to set up a new template, make sure all initial GitHub Actions have passed
2. Navigate to the repository settings and go to "Code and automation", "Pages"
3. Make sure the "Source" dropdown is set to "Deploy from a branch"
4. Set the "Branch" dropdown to "gh-pages", make sure the folder is set to "/ (root)", and click "Save"
5. Go back to the main page of your repository and click the gear to the right of the "About" section in the right side column
6. Under "Website" check the checkbox that says "Use your GitHub Pages website" and click "Save changes"
7. You will now see a URL in the "About" section that will link to your package's documentation

### 🛡️ Set Up Branch Protection (Public Repos Only)

1. Follow the Usage steps above to set up a new template, make sure all initial GitHub Actions have passed
2. Navigate to the repository settings and go to "Code and automation", "Branches"
3. Click "Add branch protection rule" to start adding branch protection
4. Under "Branch name pattern" put `main`, this will add protection to the main branch
5. Make sure to set the following options:
   - Check the "Require a pull request before merging"
   - Check the "Request status checks to pass before merging" and make sure the following status checks are added to the required list:
     - CI / Documentation
     - CI / Julia 1 - ubuntu-latest - x64 - push
     - CI / Julia 1 - ubuntu-latest - x86 - push
     - CI / Julia 1 - windows-latest - x64 - push
     - CI / Julia 1 - windows-latest - x86 - push
     - CI / Julia 1 - macOS-latest - x64 - push
   - Check the "Restrict who can push to matching branches" and add `algebraicjuliabot` to the list of people with push access
6. Click "Save changes" to enable the branch protection
