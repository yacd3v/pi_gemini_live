#!/usr/bin/env python3
"""
VocalGem Web UI (Simple Version)
A Flask-based web interface for managing VocalGem wake word detection service
This version works without sudo but shows warnings for commands that need elevated privileges
"""

import os
import subprocess
import threading
import time
import json
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, Response

# Try to import SocketIO, fallback to basic Flask if not available
try:
    from flask_socketio import SocketIO, emit
    SOCKETIO_AVAILABLE = True
except ImportError:
    SOCKETIO_AVAILABLE = False
    print("⚠️  Flask-SocketIO not available. Live log streaming will be disabled.")
    print("   Install with: pip install flask-socketio")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'vocalgem-secret-key-2024'

if SOCKETIO_AVAILABLE:
    socketio = SocketIO(app, cors_allowed_origins="*")
    # Global variables for log streaming
    log_threads = {}
    log_active = {}

def run_command(command, timeout=30):
    """Run a system command and return the result"""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return {
            'success': result.returncode == 0,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'returncode': result.returncode,
            'needs_sudo': 'sudo' in command and result.returncode != 0
        }
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'stdout': '',
            'stderr': f'Command timed out after {timeout} seconds',
            'returncode': -1,
            'needs_sudo': 'sudo' in command
        }
    except Exception as e:
        return {
            'success': False,
            'stdout': '',
            'stderr': str(e),
            'returncode': -1,
            'needs_sudo': 'sudo' in command
        }

def stream_logs(session_id, command):
    """Stream logs in real-time via WebSocket (if available)"""
    if not SOCKETIO_AVAILABLE:
        return
        
    try:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        while log_active.get(session_id, False) and process.poll() is None:
            line = process.stdout.readline()
            if line:
                socketio.emit('log_line', {
                    'line': line.strip(),
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }, room=session_id)
            time.sleep(0.1)
        
        process.terminate()
        socketio.emit('log_end', {}, room=session_id)
        
    except Exception as e:
        socketio.emit('log_error', {'error': str(e)}, room=session_id)
    finally:
        if session_id in log_threads:
            del log_threads[session_id]
        if session_id in log_active:
            del log_active[session_id]

@app.route('/')
def index():
    """Main dashboard"""
    return render_template('index_simple.html', socketio_available=SOCKETIO_AVAILABLE)

@app.route('/service/<action>')
def service_action(action):
    """Handle service management actions"""
    commands = {
        'start': 'sudo systemctl start vocalgem',
        'stop': 'sudo systemctl stop vocalgem',
        'restart': 'sudo systemctl restart vocalgem',
        'status': 'systemctl status vocalgem',
        'enable': 'sudo systemctl enable vocalgem',
        'disable': 'sudo systemctl disable vocalgem',
        'is-enabled': 'systemctl is-enabled vocalgem',
        'is-active': 'systemctl is-active vocalgem'
    }
    
    if action not in commands:
        return jsonify({'success': False, 'error': 'Invalid action'})
    
    result = run_command(commands[action])
    return jsonify(result)

@app.route('/logs/<log_type>')
def get_logs(log_type):
    """Get various types of logs"""
    commands = {
        'recent': 'journalctl -u vocalgem -n 50 --no-pager',
        'today': 'journalctl -u vocalgem --since today --no-pager',
        '10min': 'journalctl -u vocalgem --since "10 minutes ago" --no-pager',
        'errors': 'journalctl -u vocalgem --no-pager | grep -i error',
        'wake': 'journalctl -u vocalgem --no-pager | grep -i "wake word"',
        'led': 'journalctl -u vocalgem --no-pager | grep -i led'
    }
    
    if log_type not in commands:
        return jsonify({'success': False, 'error': 'Invalid log type'})
    
    result = run_command(commands[log_type])
    return jsonify(result)

@app.route('/test/<test_type>')
def run_test(test_type):
    """Run various tests"""
    commands = {
        'service': './check_service_setup.sh',
        'boot': 'sudo ./test_boot_simulation.sh',
        'led': 'python3 test_wake_led.py',
        'audio-list': 'aplay -l && arecord -l'
    }
    
    if test_type not in commands:
        return jsonify({'success': False, 'error': 'Invalid test type'})
    
    result = run_command(commands[test_type], timeout=60)
    return jsonify(result)

@app.route('/audio/<action>')
def audio_action(action):
    """Handle audio-related actions"""
    commands = {
        'devices': 'aplay -l && echo "=== RECORDING DEVICES ===" && arecord -l',
        'usb-check': 'lsusb | grep 2886',
        'reset-usb': 'sudo modprobe -r snd_usb_audio && sudo modprobe snd_usb_audio',
        'test-record': 'timeout 3 arecord -D plughw:1,0 -c 6 -r 16000 -f S16_LE test_ui.wav',
        'groups': 'groups $USER'
    }
    
    if action not in commands:
        return jsonify({'success': False, 'error': 'Invalid action'})
    
    result = run_command(commands[action])
    return jsonify(result)

@app.route('/diagnostics')
def diagnostics():
    """Run comprehensive diagnostics"""
    checks = [
        ('Service Status', 'systemctl is-active vocalgem'),
        ('Service Enabled', 'systemctl is-enabled vocalgem'),
        ('Audio Device', 'aplay -l | grep -q ReSpeaker && echo "✅ ReSpeaker detected" || echo "❌ ReSpeaker not found"'),
        ('Virtual Environment', '[ -d venv ] && echo "✅ Virtual env exists" || echo "❌ Virtual env missing"'),
        ('USB Device', 'lsusb | grep 2886 && echo "✅ USB device detected" || echo "❌ USB device not found"')
    ]
    
    results = []
    for name, command in checks:
        result = run_command(command)
        results.append({
            'name': name,
            'success': result['success'],
            'output': result['stdout'].strip() if result['stdout'] else result['stderr'].strip(),
            'status': '✅' if result['success'] else '❌',
            'needs_sudo': result.get('needs_sudo', False)
        })
    
    return jsonify({'diagnostics': results})

# SocketIO routes (only if available)
if SOCKETIO_AVAILABLE:
    @socketio.on('connect')
    def handle_connect():
        """Handle WebSocket connection"""
        print(f"Client connected: {request.sid}")

    @socketio.on('disconnect')
    def handle_disconnect():
        """Handle WebSocket disconnection"""
        session_id = request.sid
        if session_id in log_active:
            log_active[session_id] = False
        print(f"Client disconnected: {session_id}")

    @socketio.on('start_log_stream')
    def handle_start_log_stream(data):
        """Start streaming logs"""
        session_id = request.sid
        log_type = data.get('log_type', 'live')
        
        # Stop any existing stream for this session
        if session_id in log_active:
            log_active[session_id] = False
        
        # Start new stream
        log_active[session_id] = True
        
        commands = {
            'live': 'journalctl -u vocalgem -f',
            'recent': 'journalctl -u vocalgem -n 100 --no-pager',
            'today': 'journalctl -u vocalgem --since today --no-pager'
        }
        
        command = commands.get(log_type, commands['live'])
        
        # Start streaming in a separate thread
        thread = threading.Thread(target=stream_logs, args=(session_id, command))
        log_threads[session_id] = thread
        thread.start()

    @socketio.on('stop_log_stream')
    def handle_stop_log_stream():
        """Stop streaming logs"""
        session_id = request.sid
        if session_id in log_active:
            log_active[session_id] = False

if __name__ == '__main__':
    print("🚀 Starting VocalGem Web UI (Simple Version)...")
    
    if not SOCKETIO_AVAILABLE:
        print("⚠️  Note: Running without Flask-SocketIO. Live log streaming unavailable.")
        print("   To enable: pip install flask-socketio")
    
    print("🌐 Access the interface at:")
    print("   - Local: http://localhost:5000")
    print("   - Network: http://[your-pi-ip]:5000")
    print("   - Find Pi IP: hostname -I | cut -d' ' -f1")
    print("")
    print("💡 Some commands may require elevated privileges.")
    print("   For full functionality, use: sudo ./start_web_ui_sudo.sh")
    
    # Run the appropriate server
    if SOCKETIO_AVAILABLE:
        socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
    else:
        app.run(host='0.0.0.0', port=5000, debug=False) 