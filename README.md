# horguesc

A comprehensive system for horse racing data analysis and prediction, combining Python analytics with a C++/CLI data loading utility.

## Overview

horguesc provides an integrated solution for horse racing analytics, consisting of:
- A Python CLI tool for machine learning and predictions
- A C++/CLI dataloader utility for accessing JRA (Japan Racing Association) data
- A PostgreSQL database for storing and managing racing information

## System Requirements

- Windows OS (required for the JV-Link component)
- PostgreSQL database
- JRA's JV-Link data service subscription

## Installation

You have two options for installation:

### Option 1: Using the Distribution Package (Recommended)

This option is simpler and doesn't require any development tools:

1. Download the latest release package from the releases page
2. Extract it to a directory of your choice
3. Edit the `horguesc.ini` configuration file
4. Run the executables directly

The package contains:
- `horguesc.exe` - Main Python analysis tool
- `dataloader.exe` - C++ data loading utility that interfaces with JV-Link
- Supporting libraries and configuration files
- `horguesc.ini` - Configuration file (must be edited before use)

### Option 2: Building from Source

Building from source requires:
- Visual Studio (for the C++ component)
- Python development environment
- NVIDIA GPU with CUDA support (for optimal performance)

Follow these steps:

1. Clone the repository
   ```bash
   git clone https://github.com/yourusername/horguesc.git
   cd horguesc
   ```

2. Install PyTorch with CUDA support
   ```bash
   # Install PyTorch with CUDA (check https://pytorch.org for the latest command)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
   ```

3. Install remaining dependencies
   ```bash
   pip install -r requirements.txt
   ```

4. Build the C++ data loader
   - Open horguesc.sln in Visual Studio
   - Build the dataloader project in Release mode

5. Install the Python package in development mode
   ```bash
   pip install -e .
   ```

## Usage

### Data Loader Utility

```bash
# Initialize the database (creates necessary tables)
dataloader.exe setup

# Update with latest race data
dataloader.exe update

# Fetch real-time data
dataloader.exe realtime
```

The dataloader connects to JRA's data service and processes various record types:
- Race information (RA)
- Race results (SE)
- Horse information (UM)
- Jockey information (KS)
- Trainer information (CH)
- Horse pedigree (HN)
- Course information (CS)

### Analytics CLI

```bash
# Training a model
horguesc train

# Testing a model
horguesc test

# Making predictions
horguesc predict
```

## Configuration

Configure the system by editing the `horguesc.ini` file:

```ini
[database]
Host = localhost
Port = 5432
Username = postgres
Password = yourpassword
Database = horguesc

[dataloader]
Sid = YOUR_JV_LINK_SID
StartYear = 2020
DeleteDatabase = false
```

## Development

### Testing

```bash
# Run all tests
pytest

# Run specific test module
pytest tests/test_trainer.py

# Run with coverage report
pytest --cov=horguesc
```

### Building Distribution Package

```bash
# First build the dataloader project in Visual Studio
# Then build the horguesc distribution package
pyinstaller cli.spec

# The complete package will be available in dist/horguesc/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
