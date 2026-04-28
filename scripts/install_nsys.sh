#!/usr/bin/env bash
set -euo pipefail

BOLD='\033[1m'
CYAN='\033[1;36m'
GREEN='\033[1;32m'
RED='\033[1;31m'
NC='\033[0m'

step() { echo -e "${CYAN}${BOLD}>>> $1${NC}"; }
ok()   { echo -e "${GREEN}${BOLD}>>> $1${NC}"; }
err()  { echo -e "${RED}${BOLD}>>> $1${NC}"; }

if [ "$(id -u)" -ne 0 ]; then
    err "This script must be run as root."
    exit 1
fi

NSYS_URL="${NSYS_URL:-https://developer.download.nvidia.com/devtools/nsight-systems/NsightSystems-linux-public-2023.4.1.97-3355750.run}"
NSYS_ROOT="${NSYS_ROOT:-/opt/nvidia/nsight-systems/2023.4.1}"

INSTALLER_NAME="$(basename "$NSYS_URL")"
INSTALLER_PATH="/tmp/${INSTALLER_NAME}"

step "Nsight Systems installer: $NSYS_URL"
step "Target directory: $NSYS_ROOT"

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y wget ca-certificates expect

if [ -f "$INSTALLER_PATH" ]; then
    step "Installer already exists at $INSTALLER_PATH, reusing it."
else
    step "Downloading Nsight Systems installer..."
    wget -q --show-progress "$NSYS_URL" -O "$INSTALLER_PATH"
fi

chmod +x "$INSTALLER_PATH"
mkdir -p "$NSYS_ROOT"

EXPECT_SCRIPT="$(mktemp)"

cat > "$EXPECT_SCRIPT" <<'EXPECT_EOF'
#!/usr/bin/expect -f
set timeout -1

set installer $env(INSTALLER_PATH)
set target    $env(NSYS_ROOT)

spawn $installer --target $target

expect {
    -re "Press <Enter> or <Return> to read end user license agreement.*" {
        send "\r"
        exp_continue
    }

    -re "--More--\\(.*%\\)" {
        send " "
        exp_continue
    }

    -re "ACCEPT/DECLINE/QUIT" {
        send "ACCEPT\r"
        exp_continue
    }

    -re "Enter install path:.*" {
        send "\r"
        exp_continue
    }

    eof {
    }
}
EXPECT_EOF

chmod +x "$EXPECT_SCRIPT"

export INSTALLER_PATH
export NSYS_ROOT

"$EXPECT_SCRIPT"

rm -f "$EXPECT_SCRIPT"
rm -f "$INSTALLER_PATH"

NSYS_BIN="$(find "$NSYS_ROOT" -maxdepth 6 -type f -name nsys | head -n 1 || true)"

if [ -z "$NSYS_BIN" ]; then
    err "Could not find nsys under $NSYS_ROOT."
    exit 1
fi

ln -sf "$NSYS_BIN" /usr/local/bin/nsys

if command -v nsys >/dev/null 2>&1; then
    ok "nsys is available:"
    nsys --version || true
else
    err "nsys is not in PATH after symlink."
    exit 1
fi

ok "Nsight Systems installation complete."