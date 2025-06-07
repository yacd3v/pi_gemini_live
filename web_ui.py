#!/usr/bin/env python3
"""
VocalGem Web UI
A Flask-based web interface for managing VocalGem wake word detection service
"""

import os
import subprocess
import threading
import time
import json
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, Response
from flask_socketio import SocketIO, emit

app = Flask(__name__)
app.config['SECRET_KEY'] = 'vocalgem-secret-key-2024'
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
            'returncode': result.returncode
        }
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'stdout': '',
            'stderr': f'Command timed out after {timeout} seconds',
            'returncode': -1
        }
    except Exception as e:
        return {
            'success': False,
            'stdout': '',
            'stderr': str(e),
            'returncode': -1
        }

def stream_logs(session_id, command):
    """Stream logs in real-time via WebSocket"""
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
    return render_template('index.html')

@app.route('/service/<action>')
def service_action(action):
    """Handle service management actions"""
    commands = {
        'start': 'sudo systemctl start vocalgem',
        'stop': 'sudo systemctl stop vocalgem',
        'restart': 'sudo systemctl restart vocalgem',
        'status': 'sudo systemctl status vocalgem',
        'enable': 'sudo systemctl enable vocalgem',
        'disable': 'sudo systemctl disable vocalgem',
        'is-enabled': 'sudo systemctl is-enabled vocalgem',
        'is-active': 'sudo systemctl is-active vocalgem'
    }
    
    if action not in commands:
        return jsonify({'success': False, 'error': 'Invalid action'})
    
    result = run_command(commands[action])
    return jsonify(result)

@app.route('/logs/<log_type>')
def get_logs(log_type):
    """Get various types of logs"""
    commands = {
        'recent': 'sudo journalctl -u vocalgem -n 50 --no-pager',
        'today': 'sudo journalctl -u vocalgem --since today --no-pager',
        '10min': 'sudo journalctl -u vocalgem --since "10 minutes ago" --no-pager',
        'errors': 'sudo journalctl -u vocalgem --no-pager | grep -i error',
        'wake': 'sudo journalctl -u vocalgem --no-pager | grep -i "wake word"',
        'led': 'sudo journalctl -u vocalgem --no-pager | grep -i led'
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

@app.route('/troubleshoot/<action>')
def troubleshoot_action(action):
    """Handle troubleshooting actions"""
    commands = {
        'force-kill': 'sudo pkill -f wake_porcu.py',
        'force-stop': 'sudo systemctl kill vocalgem && sudo systemctl stop vocalgem',
        'gpio-cleanup': 'sudo python3 -c "import RPi.GPIO as GPIO; GPIO.cleanup()" 2>/dev/null || true',
        'complete-reset': '''
            sudo systemctl stop vocalgem;
            sudo pkill -f wake_porcu.py;
            sudo python3 -c "import RPi.GPIO as GPIO; GPIO.cleanup()" 2>/dev/null || true;
            sudo ./setup_startup.sh;
            sudo systemctl start vocalgem
        ''',
        'venv-check': 'ls -la venv/ && source venv/bin/activate && python3 -c "import pvporcupine; print(\'Porcupine OK\')" && python3 -c "import pyaudio; print(\'PyAudio OK\')"'
    }
    
    if action not in commands:
        return jsonify({'success': False, 'error': 'Invalid action'})
    
    result = run_command(commands[action], timeout=120)
    return jsonify(result)

@app.route('/diagnostics')
def diagnostics():
    """Run comprehensive diagnostics"""
    checks = [
        ('Service Status', 'sudo systemctl is-active vocalgem'),
        ('Service Enabled', 'sudo systemctl is-enabled vocalgem'),
        ('Audio Device', 'aplay -l | grep -q ReSpeaker && echo "✅ ReSpeaker detected" || echo "❌ ReSpeaker not found"'),
        ('Virtual Environment', '[ -d venv ] && echo "✅ Virtual env exists" || echo "❌ Virtual env missing"'),
        ('Wake Word Listening', 'sudo journalctl -u vocalgem --since "1 minute ago" | grep -q "Listening for wake words" && echo "✅ Listening" || echo "❌ Not listening"'),
        ('Recent Errors', 'sudo journalctl -u vocalgem --since "5 minutes ago" | grep -i error | wc -l'),
        ('USB Device', 'lsusb | grep 2886 && echo "✅ USB device detected" || echo "❌ USB device not found"')
    ]
    
    results = []
    for name, command in checks:
        result = run_command(command)
        results.append({
            'name': name,
            'success': result['success'],
            'output': result['stdout'].strip() if result['stdout'] else result['stderr'].strip(),
            'status': '✅' if result['success'] else '❌'
        })
    
    return jsonify({'diagnostics': results})

@app.route('/logs/cleanup')
def cleanup_logs():
    """Clean up old logs"""
    result = run_command('sudo journalctl --vacuum-time=1d')
    return jsonify(result)

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
        'live': 'sudo journalctl -u vocalgem -f',
        'recent': 'sudo journalctl -u vocalgem -n 100 --no-pager',
        'today': 'sudo journalctl -u vocalgem --since today --no-pager'
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
    # Check if running with correct permissions
    if os.geteuid() != 0:
        print("⚠️  Warning: Running without sudo. Some commands may fail.")
        print("For full functionality, run: sudo python3 web_ui.py")
    
    print("🚀 Starting VocalGem Web UI...")
    print("🌐 Access the interface at:")
    print("   - Local: http://localhost:5000")
    print("   - Network: http://[your-pi-ip]:5000")
    print("   - Find Pi IP: hostname -I | cut -d' ' -f1")
    
    # Run on all interfaces to allow network access
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True) 