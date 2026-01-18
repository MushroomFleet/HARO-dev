# Installing Claude Code on Steam Deck

A complete guide to installing Claude Code on your Steam Deck, including common issues and solutions.

## Prerequisites

- Steam Deck with SteamOS
- Internet connection
- Claude Pro, Claude Max subscription, or Anthropic Console account with billing enabled

## Before You Begin

**Sign into claude.ai first** — Open Firefox in Desktop Mode and sign into your Claude account at [claude.ai](https://claude.ai) before starting installation. This will speed up the authentication step later.

---

## Installation Steps

### Step 1: Switch to Desktop Mode

Hold the power button on your Steam Deck and select **Switch to Desktop Mode**.

### Step 2: Open a Terminal

Click the **Steam Deck icon** (bottom left) → **System** → **Konsole**

### Step 3: Run the Native Installer

```bash
curl -fsSL https://claude.ai/install.sh | bash
```

> **If nothing appears to happen:** The installer downloads files silently with a minimal progress indicator that may not render correctly in Konsole. The download can take several minutes depending on your connection speed. Be patient and wait for it to complete.
>
> **If you're unsure whether it's working**, download the script first, then run it with debug output:
> ```bash
> curl -fL https://claude.ai/install.sh -o install.sh
> bash -x install.sh
> ```
> This prints every command as it executes so you can see progress.

> **If curl hangs indefinitely:** Test your network connection:
> ```bash
> ping -c 3 google.com
> curl -I https://claude.ai
> ```
> If these fail, restart the network manager:
> ```bash
> sudo systemctl restart NetworkManager
> ```

### Step 4: Refresh Your Shell

After installation completes, reload your shell configuration:

```bash
source ~/.bashrc
```

Or simply close and reopen Konsole.

### Step 5: Verify Installation

```bash
claude --version
```

You should see a version number printed.

> **If you see "command not found":** The PATH may not be set correctly. Try:
> ```bash
> export PATH="$HOME/.claude/bin:$HOME/.local/bin:$PATH"
> claude --version
> ```
> If that works, add the export line to your `~/.bashrc`:
> ```bash
> echo 'export PATH="$HOME/.claude/bin:$HOME/.local/bin:$PATH"' >> ~/.bashrc
> ```

### Step 6: Authenticate

Navigate to any folder and launch Claude Code:

```bash
cd ~
claude
```

Follow the on-screen prompts to complete OAuth authentication. If you signed into claude.ai in Firefox beforehand, this step will be faster.

> **If the browser doesn't open automatically:** Copy the URL shown in the terminal and paste it into Firefox manually.

---

## Alternative Installation Method (npm)

If the native installer fails, you can use npm instead:

```bash
# Install Node.js and npm
sudo pacman -S nodejs npm

# Configure npm for user-level installs (avoids permission issues)
mkdir -p ~/.npm-global
npm config set prefix ~/.npm-global
echo 'export PATH=~/.npm-global/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

# Install Claude Code
npm install -g @anthropic-ai/claude-code

# Verify
claude --version
```

> **Never use `sudo` with npm install** — this causes permission issues and security risks.

---

## Troubleshooting

### Run the Built-in Diagnostic Tool

```bash
claude doctor
```

This checks your installation and suggests fixes for common problems.

### White Square or Garbled Output in Terminal

Konsole on Steam Deck may not render some Unicode characters correctly. This is cosmetic and doesn't affect functionality. If you're unsure whether a command is running:

```bash
bash install.sh 2>&1 | cat
```

### Script Downloads HTML Instead of Bash

Check the contents of a downloaded script before running:

```bash
head -20 install.sh
```

If you see HTML tags, the download failed or was redirected. Try an alternative URL or the npm method.

### Stuck at Authentication

- Make sure you're signed into claude.ai in a browser first
- Press Enter if the terminal appears frozen (prompts may not display)
- Try typing `y` and pressing Enter if it seems to be waiting for confirmation

---

## Tips for Using Claude Code on Steam Deck

- **Use a physical keyboard** — The on-screen keyboard is tedious for terminal work
- **Storage** — Consider your available space; use an SD card if the internal drive is full
- **Performance** — Claude Code runs in the terminal and offloads processing to Anthropic's servers, so the Steam Deck handles it fine
- **Return to Gaming Mode** — When you're done, click the "Return to Gaming Mode" shortcut on the desktop

---

## Uninstalling

If you need to remove Claude Code:

```bash
# Remove the binary
rm -rf ~/.claude
rm -f ~/.local/bin/claude

# Remove configuration (optional)
rm -rf ~/.claude.json
```

For npm installations:

```bash
npm uninstall -g @anthropic-ai/claude-code
```

---

## Feedback

If you encounter bugs or issues with Claude Code:

- Use the `/bug` command inside Claude Code
- File an issue at [github.com/anthropics/claude-code](https://github.com/anthropics/claude-code/issues)
