# Contributing to Legion

We're excited that you're interested in contributing to Legion! This guide will help you get started with the contribution process.

## Ways to Contribute

- **Bug Reports**: If you find a bug, please create an issue in our GitHub repository
- **Feature Requests**: Have an idea for a new feature? Share it through GitHub issues
- **Documentation**: Help improve our documentation by fixing typos, adding examples, or clarifying explanations
- **Code Contributions**: Submit pull requests for bug fixes or new features

## Development Setup

For information about setting up your development environment, please refer to our installation guide.

## Pull Request Process

1. **Create Pull Request**
   - Push your changes to your fork
   - Create a PR against the `main` branch
   - Fill out the PR template
   - Link any related issues

2. **PR Guidelines**
   - Keep changes focused and atomic
   - Provide clear description of changes
   - Include any necessary documentation updates
   - Add screenshots for UI changes
   - Reference any related issues

3. **Review Process**
   - Maintainers will review your PR
   - Address any requested changes
   - Once approved, maintainers will merge your PR

### **Collaboration Patterns**

1. **Direct to Upstream** (Recommended)
   - Each contributor maintains their fork
   - Create PRs directly to upstream
   - Sync fork:
     ```bash
     git fetch upstream
     git rebase upstream/main
     ```

2. **Fork Collaboration**
   - Add collaborators to your fork's settings
   - Both can push branches
   - Create single PR to upstream
   - Maintain clear ownership of the PR

3. **Cross-fork PRs**
   - For non-collaborator contributions
   - Create PR between forks
   - Owner submits final PR upstream

Note: Contributions are tracked through PRs, not individual commits due to squash merging. All contributors are credited in the PR history.

## Code Style

- Follow the existing code style
- Include comments where necessary
- Write clear commit messages

## Questions?

If you have questions, feel free to open a discussion on GitHub.
