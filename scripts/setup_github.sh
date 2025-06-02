#!/bin/bash

# =============================================================================
# GitHub Setup Script for nanoNAS
# =============================================================================
#
# This script helps set up the nanoNAS project on GitHub with proper
# repository initialization, branch setup, and initial push.
#
# Usage: ./scripts/setup_github.sh [repository_url]
#
# =============================================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if git is installed
check_git() {
    if ! command -v git &> /dev/null; then
        log_error "Git is not installed. Please install Git first."
        exit 1
    fi
    log_success "Git is available"
}

# Check if we're in the right directory
check_project_directory() {
    if [ ! -f "setup.py" ] || [ ! -d "nanonas" ]; then
        log_error "This doesn't appear to be the nanoNAS project directory."
        log_error "Please run this script from the project root."
        exit 1
    fi
    log_success "Project directory verified"
}

# Initialize git repository if not already done
init_git() {
    if [ ! -d ".git" ]; then
        log_info "Initializing Git repository..."
        git init
        log_success "Git repository initialized"
    else
        log_info "Git repository already exists"
    fi
}

# Set up gitignore and other files
setup_files() {
    log_info "Setting up repository files..."
    
    # Ensure all necessary files exist
    touch README.md
    touch .gitignore
    
    log_success "Repository files ready"
}

# Add and commit files
commit_files() {
    log_info "Adding files to Git..."
    
    # Add all files
    git add .
    
    # Check if there are changes to commit
    if git diff --staged --quiet; then
        log_warning "No changes to commit"
        return
    fi
    
    # Commit with descriptive message
    git commit -m "feat: initial nanoNAS framework implementation

- Complete neural architecture search framework
- Multiple search strategies (Evolutionary, DARTS, Random)
- Comprehensive benchmarking and evaluation
- Advanced visualization capabilities
- Docker support and CI/CD pipeline
- Graduate-school level documentation

This represents a complete transformation from basic AutoML-CodeGen
to a publication-ready Neural Architecture Search framework."
    
    log_success "Files committed to Git"
}

# Set up remote repository
setup_remote() {
    local repo_url="$1"
    
    if [ -z "$repo_url" ]; then
        log_warning "No repository URL provided."
        log_info "Please create a GitHub repository first, then run:"
        log_info "git remote add origin <your-repo-url>"
        log_info "git push -u origin main"
        return
    fi
    
    log_info "Setting up remote repository: $repo_url"
    
    # Add remote if not exists
    if ! git remote get-url origin &> /dev/null; then
        git remote add origin "$repo_url"
        log_success "Remote origin added"
    else
        log_warning "Remote origin already exists"
    fi
}

# Create and switch to main branch
setup_branches() {
    log_info "Setting up branches..."
    
    # Check current branch
    current_branch=$(git branch --show-current 2>/dev/null || echo "")
    
    if [ "$current_branch" != "main" ]; then
        # Rename current branch to main if it exists, otherwise create main
        if [ -n "$current_branch" ]; then
            git branch -M main
        else
            git checkout -b main
        fi
        log_success "Main branch created/renamed"
    else
        log_info "Already on main branch"
    fi
    
    # Create develop branch
    if ! git show-ref --verify --quiet refs/heads/develop; then
        git checkout -b develop main
        git checkout main
        log_success "Develop branch created"
    else
        log_info "Develop branch already exists"
    fi
}

# Push to GitHub
push_to_github() {
    local repo_url="$1"
    
    if [ -z "$repo_url" ]; then
        log_warning "Skipping push - no repository URL provided"
        return
    fi
    
    log_info "Pushing to GitHub..."
    
    # Push main branch
    if git push -u origin main; then
        log_success "Main branch pushed to GitHub"
    else
        log_error "Failed to push main branch"
        log_info "You may need to authenticate with GitHub or check the repository URL"
        return 1
    fi
    
    # Push develop branch
    if git push -u origin develop; then
        log_success "Develop branch pushed to GitHub"
    else
        log_warning "Failed to push develop branch (this is usually okay)"
    fi
}

# Create GitHub-specific files
create_github_files() {
    log_info "Creating GitHub-specific files..."
    
    # Ensure .github directories exist
    mkdir -p .github/workflows
    mkdir -p .github/ISSUE_TEMPLATE
    
    log_success "GitHub directories created"
}

# Main execution
main() {
    local repo_url="$1"
    
    echo "🚀 Setting up nanoNAS on GitHub"
    echo "================================"
    echo ""
    
    # Run setup steps
    check_git
    check_project_directory
    init_git
    setup_files
    create_github_files
    setup_branches
    commit_files
    setup_remote "$repo_url"
    
    if [ -n "$repo_url" ]; then
        push_to_github "$repo_url"
    fi
    
    echo ""
    echo "🎉 GitHub setup complete!"
    echo ""
    
    if [ -n "$repo_url" ]; then
        log_success "Repository is now available at: $repo_url"
    else
        log_info "Next steps:"
        log_info "1. Create a new repository on GitHub"
        log_info "2. Run: git remote add origin <your-repo-url>"
        log_info "3. Run: git push -u origin main"
    fi
    
    echo ""
    log_info "GitHub Actions CI/CD will run automatically on push"
    log_info "Check the Actions tab in your GitHub repository"
    echo ""
    log_info "Happy coding! 🔬"
}

# Run main function with arguments
main "$@" 