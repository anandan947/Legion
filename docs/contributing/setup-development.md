# Development Environment Setup

This guide provides comprehensive instructions for setting up your development environment for Legion.

## Development Environment Options

Legion offers two primary methods for setting up your development environment:

1. Manual Installation
2. Nix Shell (Recommended for team consistency)

## Manual Installation

### Prerequisites

- Windows 10/11 with WSL2 enabled
- Ubuntu on WSL2
- Git

### Step-by-Step Setup

1. Update WSL Ubuntu System
```bash
sudo apt update && sudo apt upgrade -y
```

2. Install Python and Development Tools
```bash
sudo apt install -y python3.11 python3.11-venv python3-pip build-essential
```

3. Create and Activate Virtual Environment
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

4. Install Python Dependencies
```bash
pip install -r requirements.txt
```

## Nix Shell Setup

Nix provides a fully reproducible development environment that ensures all team members have identical setups.

### What is Nix?

Nix creates isolated development environments that bundle everything your project needs - from system packages to language-specific dependencies. Unlike traditional approaches, Nix manages all dependencies in isolation, ensuring consistent development environments across your team.

### Installation Steps

1. Install Nix Package Manager
```bash
sh <(curl -L https://nixos.org/nix/install) --daemon
```

2. Enable Flakes (Optional)
Add to `~/.config/nix/nix.conf`:
```
experimental-features = nix-command flakes
```

3. Using the Development Environment
```bash
nix-shell shell.nix
```

### Troubleshooting Nix Setup

#### Permission Issues
- Verify group membership: `groups`
- Check socket permissions: `ls -l /nix/var/nix/daemon-socket/socket`
- Restart daemon if needed: `sudo systemctl restart nix-daemon`

#### Channel Update Failures
- Check internet connection
- Verify channel: `nix-channel --list`
- Try removing and re-adding channel

## Environment Verification

To verify your setup is working correctly:

1. Activate the environment (either venv or nix-shell)
2. Run the verification script:
```bash
python verify_env.py
```

## Included Tools and Dependencies

- Python 3.11
- pip
- venv
- build-essential
- Development libraries

## Maintenance

Keep your development environment up to date:

1. Regular Updates
```bash
git pull origin main
pip install -r requirements.txt  # For manual setup
# OR
nix-shell  # For Nix setup (will automatically update)
```

2. Verify Environment
```bash
python verify_env.py
```

## Need Help?

If you encounter any issues during setup:

1. Check our [Troubleshooting Guide](troubleshooting.md)
2. Open an issue on our [GitHub repository](https://github.com/LLMP-io/Legion)
3. Join our [Discord community](https://discord.gg/legion) for real-time support
