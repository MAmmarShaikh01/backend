# Setting up Virtual Environment to Avoid Pip Root User Warning

To avoid the warning about running pip as the 'root' user, please follow these steps to set up a virtual environment:

## Quick Setup

Run the setup script to create a virtual environment:

```bash
python setup_venv.py
```

## Manual Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

3. Upgrade pip in the virtual environment:
   ```bash
   pip install --upgrade pip
   ```

4. Install the project dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. When you're done working, deactivate the virtual environment:
   ```bash
   deactivate
   ```

## Running the Application

After activating the virtual environment:

```bash
# Activate virtual environment first
source venv/bin/activate  # On Linux/macOS
# or
venv\Scripts\activate     # On Windows

# Install dependencies if not already done
pip install -r requirements.txt

# Run the application
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Why Use a Virtual Environment?

Using a virtual environment:
- Prevents the "running pip as root" warning
- Avoids conflicts between project dependencies
- Keeps your system Python installation clean
- Ensures consistent dependency versions across environments