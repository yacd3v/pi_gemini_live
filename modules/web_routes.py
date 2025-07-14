#!/usr/bin/env python3
"""
Web routes module for robot control dashboard
Handles all Flask routes and API endpoints
"""

import time
from flask import render_template, Response, jsonify, request

class WebRoutes:
    """Manages all web routes and API endpoints"""
    
    def __init__(self, app, managers):
        self.app = app
        self.managers = managers
        self.setup_routes()
    
    def setup_routes(self):
        """Setup all Flask routes"""
        
        @self.app.route('/')
        def index():
            """Main page with video stream, IMU visualization, and motor controls"""
            return render_template('index.html')
        
        @self.app.route('/video_feed')
        def video_feed():
            """Ultra-low latency video streaming route"""
            return Response(
                self.managers['camera'].generate_frames(),
                mimetype='multipart/x-mixed-replace; boundary=frame',
                headers={
                    'Cache-Control': 'no-cache, no-store, max-age=0',
                    'Pragma': 'no-cache',
                    'Expires': '0',
                    'X-Accel-Buffering': 'no',  # Disable nginx buffering
                    'Connection': 'keep-alive'
                }
            )
        
        @self.app.route('/imu_data')
        def imu_data_endpoint():
            """IMU data endpoint for AJAX requests"""
            # Get data from all managers
            imu_data = self.managers['imu'].get_data()
            ultrasonic_distance = self.managers['ultrasonic'].get_distance()
            battery_data = self.managers['battery'].get_data()
            adc_data = self.managers['adc'].get_data()
            servo_angles = self.managers['servo'].get_angles()
            
            # Combine all data
            combined_data = {
                **imu_data,
                'ultrasonic_distance': ultrasonic_distance,
                **battery_data,
                **adc_data,
                'servo_pan': servo_angles['pan'],
                'servo_tilt': servo_angles['tilt'],
                'status': 'online' if self.managers['imu'].bno else 'offline',
                'imu_available': self.managers['imu'].get_status()['available'],
                'ultrasonic_available': self.managers['ultrasonic'].get_status()['available'],
                'battery_available': self.managers['battery'].get_status()['available'],
                'adc_available': self.managers['adc'].get_status()['available'],
                'servo_available': self.managers['servo'].get_status()['available'],
                'audio_available': self.managers['audio'].get_status()['available']
            }
            
            return jsonify(combined_data)
        
        @self.app.route('/motor_control', methods=['POST'])
        def motor_control():
            """Motor control endpoint for web interface"""
            try:
                data = request.get_json()
                action = data.get('action')
                
                if action == 'enable':
                    success, message = self.managers['motor'].enable_motors()
                    return jsonify({'success': success, 'message': message})
                    
                elif action == 'disable':
                    success, message = self.managers['motor'].disable_motors()
                    return jsonify({'success': success, 'message': message})
                    
                elif action == 'stop':
                    success, message = self.managers['motor'].emergency_stop()
                    return jsonify({'success': success, 'message': message})
                    
                elif action == 'move':
                    direction = data.get('direction')
                    speed = data.get('speed', 2000)
                    success, message = self.managers['motor'].move(direction, speed)
                    return jsonify({'success': success, 'message': message})
                    
                else:
                    return jsonify({'success': False, 'error': 'Invalid action'})
                    
            except Exception as e:
                print(f"Motor control error: {e}")
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/navigation_control', methods=['POST'])
        def navigation_control():
            """Navigation control endpoint for autonomous navigation"""
            try:
                data = request.get_json()
                action = data.get('action')
                
                if action == 'navigate':
                    angle = data.get('angle', 0)
                    distance = data.get('distance', 0.5)
                    
                    if 'navigation' not in self.managers:
                        return jsonify({'success': False, 'error': 'Navigation not available'})
                    
                    success, message = self.managers['navigation'].navigate_to_angle_distance(angle, distance)
                    return jsonify({'success': success, 'message': message})
                    
                elif action == 'stop':
                    if 'navigation' not in self.managers:
                        return jsonify({'success': False, 'error': 'Navigation not available'})
                    
                    success, message = self.managers['navigation'].stop_navigation()
                    return jsonify({'success': success, 'message': message})
                    
                else:
                    return jsonify({'success': False, 'error': 'Invalid action'})
                    
            except Exception as e:
                print(f"Navigation control error: {e}")
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/navigation_status')
        def navigation_status():
            """Navigation status endpoint for AJAX requests"""
            try:
                if 'navigation' not in self.managers:
                    return jsonify({'available': False})
                
                nav_status = self.managers['navigation'].get_navigation_status()
                system_status = self.managers['navigation'].get_status()
                
                return jsonify({
                    'available': True,
                    'system_status': system_status,
                    'navigation_status': nav_status
                })
                
            except Exception as e:
                print(f"Navigation status error: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({'available': False, 'error': str(e)})
        
        @self.app.route('/toggle_overlays', methods=['POST'])
        def toggle_overlays():
            """Toggle overlay processing for debugging latency"""
            overlays_enabled = self.managers['camera'].toggle_overlays()
            return jsonify({'success': True, 'overlays_enabled': overlays_enabled})
        
        @self.app.route('/servo_control', methods=['POST'])
        def servo_control():
            """Servo control endpoint for camera pan/tilt"""
            try:
                data = request.get_json()
                action = data.get('action')
                
                if action == 'pan':
                    angle = data.get('angle', 90)
                    success, message = self.managers['servo'].set_pan(angle)
                    if success:
                        return jsonify({'success': True, 'message': message, 'angle': angle})
                    else:
                        return jsonify({'success': False, 'error': message})
                    
                elif action == 'tilt':
                    angle = data.get('angle', 90)
                    success, message = self.managers['servo'].set_tilt(angle)
                    if success:
                        return jsonify({'success': True, 'message': message, 'angle': angle})
                    else:
                        return jsonify({'success': False, 'error': message})
                    
                elif action == 'center':
                    success, message = self.managers['servo'].center_camera()
                    if success:
                        return jsonify({'success': True, 'message': message, 'pan': 90, 'tilt': 90})
                    else:
                        return jsonify({'success': False, 'error': message})
                    
                else:
                    return jsonify({'success': False, 'error': 'Invalid action'})
                    
            except Exception as e:
                print(f"Servo control error: {e}")
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/audio_data')
        def audio_data_endpoint():
            """Audio data endpoint for AJAX requests"""
            # Get audio data from manager
            audio_data = self.managers['audio'].get_audio_data()
            doa_data = self.managers['audio'].get_doa_visualization_data()
            
            return jsonify({
                'doa_angle': float(doa_data['angle']),
                'doa_x': float(doa_data['x']),
                'doa_y': float(doa_data['y']),
                'voice_activity': bool(doa_data['voice_activity']),
                'volume_level': float(doa_data['volume_level']),
                'volume_db': float(doa_data['volume_db']),
                'timestamp': float(doa_data['timestamp'])
            })
        
        @self.app.route('/audio_spectrum')
        def audio_spectrum_endpoint():
            """Audio spectrum endpoint for visualization"""
            spectrum = self.managers['audio'].get_audio_spectrum()
            if spectrum:
                return jsonify(spectrum)
            else:
                return jsonify({'error': 'No audio spectrum available'})
        
        @self.app.route('/status')
        def status():
            """Status endpoint with performance and system info"""
            camera_stats = self.managers['camera'].get_stats()
            
            return {
                'status': 'running',
                'camera': 'IMX500' if self.managers['camera'].camera else 'not initialized',
                'imu': self.managers['imu'].get_status(),
                'ultrasonic': self.managers['ultrasonic'].get_status(),
                'battery': self.managers['battery'].get_status(),
                'adc': self.managers['adc'].get_status(),
                'servos': self.managers['servo'].get_status(),
                'motors': self.managers['motor'].get_status(),
                'audio': self.managers['audio'].get_status(),
                'config': self.managers['camera'].config,
                'frame_stats': camera_stats['frame_stats'],
                'optimization_mode': 'ultra-low-latency',
                'overlays_enabled': not self.managers['camera'].config.get('skip_overlays', True),
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            }
        
        @self.app.route('/robot_mode/<action>', methods=['POST'])
        def robot_mode_control(action):
            """Control robot mode activation/deactivation with audio server management"""
            try:
                if action == 'enable':
                    # Enable robot mode - audio manager should already be handling PipeWire
                    success = True
                    message = "Robot mode enabled - exclusive audio access active"
                    
                elif action == 'disable':
                    # Disable robot mode - audio servers will be restored on cleanup
                    # For immediate restoration without full cleanup:
                    if hasattr(self.managers['audio'], '_manage_audio_servers'):
                        self.managers['audio']._manage_audio_servers('restore')
                    success = True
                    message = "Robot mode disabled - desktop audio restored"
                    
                else:
                    success = False
                    message = f"Invalid action: {action}"
                
                return jsonify({
                    'success': success,
                    'message': message,
                    'action': action,
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                })
                
            except Exception as e:
                return jsonify({
                    'success': False,
                    'message': f"Failed to {action} robot mode: {str(e)}",
                    'action': action,
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                })
        
        @self.app.route('/audio_control/<action>', methods=['POST'])
        def audio_control(action):
            """Control audio system and PipeWire management"""
            try:
                if action == 'stop_pipewire':
                    if hasattr(self.managers['audio'], '_manage_audio_servers'):
                        success = self.managers['audio']._manage_audio_servers('stop')
                        message = "Audio servers stopped for exclusive access" if success else "Failed to stop audio servers"
                    else:
                        success = False
                        message = "Audio server management not available"
                        
                elif action == 'restore_pipewire':
                    if hasattr(self.managers['audio'], '_manage_audio_servers'):
                        success = self.managers['audio']._manage_audio_servers('restore')
                        message = "Desktop audio servers restored" if success else "Failed to restore audio servers"
                    else:
                        success = False
                        message = "Audio server management not available"
                        
                else:
                    success = False
                    message = f"Invalid action: {action}"
                
                return jsonify({
                    'success': success,
                    'message': message,
                    'action': action,
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                })
                
            except Exception as e:
                return jsonify({
                    'success': False,
                    'message': f"Audio control failed: {str(e)}",
                    'action': action,
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                }) 