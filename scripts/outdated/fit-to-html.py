#!/usr/bin/env python3
"""
FIT to HTML Dream Report Generator
Genera reportes HTML clean y modernos desde archivos FIT de running
Versión optimizada con manejo robusto de datos y diseño minimalista
"""

import fitdecode
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
from pathlib import Path
import pytz

# Opcional: para detección de timezone y ubicación desde GPS
try:
    from timezonefinder import TimezoneFinder
    TIMEZONE_FINDER_AVAILABLE = True
except ImportError:
    TIMEZONE_FINDER_AVAILABLE = False

try:
    import reverse_geocoder as rg
    REVERSE_GEOCODER_AVAILABLE = True
except ImportError:
    REVERSE_GEOCODER_AVAILABLE = False

class SafeDataProcessor:
    """Clase base para procesamiento seguro de datos FIT"""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
    
    def safe_execute(self, func, *args, **kwargs):
        """Ejecuta función con manejo de errores"""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = f"Error en {func.__name__}: {str(e)}"
            self.errors.append(error_msg)
            print(f"❌ {error_msg}")
            return None
    
    def add_warning(self, message):
        """Añade warning no crítico"""
        self.warnings.append(message)
        print(f"⚠️ {message}")
    
    def safe_format_float(self, value, decimals=2, fallback='N/A'):
        """Formatea floats de manera segura"""
        if pd.isna(value) or not isinstance(value, (int, float)):
            return fallback
        return f"{value:.{decimals}f}"
    
    def safe_format_time(self, seconds, fallback='N/A'):
        """Formatea tiempo de manera segura"""
        if pd.isna(seconds) or seconds <= 0:
            return fallback
        
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes}:{secs:02d}"
    
    def safe_format_pace(self, minutes_per_km, fallback='N/A'):
        """Formatea pace de manera segura"""
        if pd.isna(minutes_per_km) or minutes_per_km <= 0:
            return fallback
        mins = int(minutes_per_km)
        secs = int((minutes_per_km % 1) * 60)
        return f"{mins}:{secs:02d}"

class FitToHtmlGenerator(SafeDataProcessor):
    def __init__(self, fit_file_path):
        super().__init__()
        self.fit_file_path = fit_file_path
        self.records = []
        self.laps = []
        self.sessions = []
        self.events = []
        self.device_info = []
        
        # DataFrames procesados
        self.df_records = None
        self.df_laps = None
        self.df_events = None
        
        # Análisis
        self.pauses = []
        self.key_devices = []
        self.local_timezone = None
        self.session_start_local = None
        self.location_info = None
        self.summary_data = {}
        self.dynamics_data = {}
        
    def load_fit_file(self):
        """Carga y procesa el archivo FIT"""
        print(f"📂 Analizando: {os.path.basename(self.fit_file_path)}")
        
        try:
            with fitdecode.FitReader(self.fit_file_path) as fit:
                for frame in fit:
                    if isinstance(frame, fitdecode.FitDataMessage):
                        self._process_message(frame)
        except Exception as e:
            self.errors.append(f"Error cargando archivo: {e}")
            return False
            
        return True
    
    def _process_message(self, frame):
        """Procesa cada mensaje del archivo FIT"""
        msg_type = frame.name
        data = {field.name: field.value for field in frame.fields}
        
        if msg_type == 'record':
            self.records.append(data)
        elif msg_type == 'lap':
            self.laps.append(data)
        elif msg_type == 'session':
            self.sessions.append(data)
        elif msg_type == 'event':
            self.events.append(data)
        elif msg_type == 'device_info':
            self.device_info.append(data)
    
    def process_data(self):
        """Convierte y procesa todos los datos"""
        # Crear DataFrames de manera segura
        self.df_records = pd.DataFrame(self.records) if self.records else pd.DataFrame()
        self.df_laps = pd.DataFrame(self.laps) if self.laps else pd.DataFrame()
        self.df_events = pd.DataFrame(self.events) if self.events else pd.DataFrame()
        
        # Pipeline de procesamiento
        self._process_timestamps()
        self._detect_local_timezone()
        self._calculate_derived_metrics()
        self._filter_invalid_laps()
        self._calculate_cumulative_metrics()
        self._detect_pauses()
        self._calculate_summary_data()
        self._calculate_dynamics_data()
        
        print(f"✅ Procesamiento completado: {len(self.errors)} errores, {len(self.warnings)} warnings")
    
    def _process_timestamps(self):
        """Procesa timestamps para análisis temporal"""
        if not self.df_records.empty and 'timestamp' in self.df_records.columns:
            self.df_records['timestamp'] = pd.to_datetime(self.df_records['timestamp'])
            if len(self.df_records) > 1:
                self.df_records['elapsed_seconds'] = (
                    self.df_records['timestamp'] - self.df_records['timestamp'].iloc[0]
                ).dt.total_seconds()
    
    def _detect_local_timezone(self):
        """Detecta timezone y ubicación desde GPS"""
        if not TIMEZONE_FINDER_AVAILABLE or self.df_records.empty:
            return
        
        if 'position_lat' in self.df_records.columns and 'position_long' in self.df_records.columns:
            valid_coords = self.df_records[
                (self.df_records['position_lat'].notna()) & 
                (self.df_records['position_long'].notna()) &
                (self.df_records['position_lat'] != 0) &
                (self.df_records['position_long'] != 0)
            ]
            
            if len(valid_coords) > 0:
                lat_semicircles = valid_coords['position_lat'].iloc[0]
                lng_semicircles = valid_coords['position_long'].iloc[0]
                
                lat = lat_semicircles * (180 / (2**31))
                lng = lng_semicircles * (180 / (2**31))
                
                try:
                    tf = TimezoneFinder()
                    timezone_str = tf.timezone_at(lat=lat, lng=lng)
                    
                    if timezone_str:
                        self.local_timezone = pytz.timezone(timezone_str)
                        
                        if self.sessions and 'start_time' in self.sessions[0]:
                            utc_time = self.sessions[0]['start_time']
                            if utc_time.tzinfo is None:
                                utc_time = pytz.utc.localize(utc_time)
                            self.session_start_local = utc_time.astimezone(self.local_timezone)
                    
                    # Detectar ubicación
                    if REVERSE_GEOCODER_AVAILABLE:
                        location_results = rg.search([(lat, lng)])
                        if location_results:
                            location = location_results[0]
                            city = location.get('name', 'Unknown')
                            admin1 = location.get('admin1', '')
                            country = location.get('cc', '')
                            self.location_info = f"{city}, {admin1}, {country}"
                            
                except Exception as e:
                    self.add_warning(f"Error detectando ubicación: {e}")
    
    def _calculate_derived_metrics(self):
        """Calcula métricas derivadas esenciales"""
        if self.df_records.empty:
            return
        
        # Corregir cadencia - multiplicar por 2
        if 'cadence' in self.df_records.columns:
            self.df_records['cadence_corrected'] = self.df_records['cadence'] * 2
        
        # Pace en min/km desde velocidad
        if 'enhanced_speed' in self.df_records.columns:
            self.df_records['pace_min_km'] = np.where(
                self.df_records['enhanced_speed'] > 0,
                1000 / (self.df_records['enhanced_speed'] * 60),
                np.nan
            )
        
        # Ground Contact Time por pie
        if 'stance_time' in self.df_records.columns and 'stance_time_balance' in self.df_records.columns:
            balance = self.df_records['stance_time_balance'] / 100.0
            self.df_records['gct_left_ms'] = self.df_records['stance_time'] * balance
            self.df_records['gct_right_ms'] = self.df_records['stance_time'] * (1 - balance)
            self.df_records['gct_imbalance_ms'] = abs(self.df_records['gct_left_ms'] - self.df_records['gct_right_ms'])
        
        # Step length en metros
        if 'step_length' in self.df_records.columns:
            self.df_records['step_length_m'] = self.df_records['step_length'] / 1000
        
        # Procesar laps también
        if not self.df_laps.empty:
            # Pace calculado por lap
            if 'total_timer_time' in self.df_laps.columns and 'total_distance' in self.df_laps.columns:
                self.df_laps['pace_calculated'] = self.df_laps.apply(
                    lambda row: (row['total_timer_time'] / 60) / (row['total_distance'] / 1000) 
                    if row['total_distance'] > 0 else np.nan, axis=1
                )
            
            # Corregir cadencia en laps
            if 'avg_running_cadence' in self.df_laps.columns:
                self.df_laps['avg_running_cadence_corrected'] = self.df_laps['avg_running_cadence'] * 2
            if 'max_running_cadence' in self.df_laps.columns:
                self.df_laps['max_running_cadence_corrected'] = self.df_laps['max_running_cadence'] * 2
            
            # Step length en metros
            if 'avg_step_length' in self.df_laps.columns:
                self.df_laps['avg_step_length_m'] = self.df_laps['avg_step_length'] / 1000
    
    def _filter_invalid_laps(self):
        """Filtra laps inválidos (duración menor a 10 segundos)"""
        if self.df_laps.empty:
            return
        
        if 'total_timer_time' in self.df_laps.columns:
            valid_laps_mask = self.df_laps['total_timer_time'] >= 10
            invalid_count = len(self.df_laps) - valid_laps_mask.sum()
            
            if invalid_count > 0:
                self.add_warning(f"Filtrando {invalid_count} lap(s) inválido(s) (< 10 segundos)")
            
            self.df_laps = self.df_laps[valid_laps_mask].reset_index(drop=True)
    
    def _calculate_cumulative_metrics(self):
        """Calcula métricas acumuladas por lap"""
        if self.df_laps.empty:
            return
        
        if 'total_timer_time' in self.df_laps.columns:
            self.df_laps['cumulative_time'] = self.df_laps['total_timer_time'].cumsum()
        
        if 'total_distance' in self.df_laps.columns:
            self.df_laps['cumulative_distance'] = self.df_laps['total_distance'].cumsum()
    
    def _detect_pauses(self):
        """Detecta pausas con distancia y duración"""
        if self.df_events.empty or self.df_records.empty:
            return
        
        timer_events = self.df_events[
            (self.df_events['event'] == 'timer') & 
            (self.df_events['event_type'].isin(['start', 'stop_all']))
        ].copy()
        
        if len(timer_events) < 2:
            return
        
        timer_events = timer_events.sort_values('timestamp').reset_index(drop=True)
        
        for i in range(len(timer_events) - 1):
            if (timer_events.iloc[i]['event_type'] == 'stop_all' and 
                timer_events.iloc[i + 1]['event_type'] == 'start'):
                
                stop_time = timer_events.iloc[i]['timestamp']
                start_time = timer_events.iloc[i + 1]['timestamp']
                duration = (start_time - stop_time).total_seconds()
                
                pause_record = self.df_records[self.df_records['timestamp'] <= stop_time]
                if not pause_record.empty and 'distance' in pause_record.columns:
                    distance_km = pause_record['distance'].iloc[-1] / 1000
                    
                    # Clasificar tipo de pausa
                    if duration < 30:
                        pause_type = "Micro-pausa"
                    elif duration < 300:
                        pause_type = "Pausa corta"
                    else:
                        pause_type = "Pausa larga"
                    
                    self.pauses.append({
                        'distance_km': distance_km,
                        'duration_seconds': duration,
                        'duration_formatted': str(timedelta(seconds=int(duration))),
                        'type': pause_type
                    })
    
    def _calculate_summary_data(self):
        """Calcula datos de resumen para el HTML"""
        self.summary_data = {
            'duration': 'N/A',
            'distance': 'N/A',
            'normalized_power': 'N/A',
            'avg_power': 'N/A',
            'max_power': 'N/A',
            'avg_hr': 'N/A',
            'max_hr': 'N/A',
            'calories': 'N/A',
            'avg_cadence': 'N/A',
            'avg_pace': 'N/A',
            'avg_step_length': 'N/A',
            'elevation_gain': 'N/A',
            'elevation_loss': 'N/A',
            'total_laps': len(self.df_laps) if not self.df_laps.empty else 0
        }
        
        # Datos de sesión
        if self.sessions:
            session = self.sessions[0]
            
            if 'total_elapsed_time' in session:
                self.summary_data['duration'] = self.safe_format_time(session['total_elapsed_time'])
            
            if 'total_distance' in session:
                distance_km = session['total_distance'] / 1000
                self.summary_data['distance'] = self.safe_format_float(distance_km, 2)
            
            if 'normalized_power' in session:
                self.summary_data['normalized_power'] = str(int(session['normalized_power']))
            
            if 'total_calories' in session:
                self.summary_data['calories'] = str(int(session['total_calories']))
        
        # Datos promedio desde records
        if not self.df_records.empty:
            if 'heart_rate' in self.df_records.columns:
                avg_hr = self.df_records['heart_rate'].mean()
                if pd.notna(avg_hr):
                    self.summary_data['avg_hr'] = str(int(avg_hr))
                
                max_hr = self.df_records['heart_rate'].max()
                if pd.notna(max_hr):
                    self.summary_data['max_hr'] = str(int(max_hr))
            
            if 'power' in self.df_records.columns:
                avg_power = self.df_records['power'].mean()
                if pd.notna(avg_power):
                    self.summary_data['avg_power'] = str(int(avg_power))
                
                max_power = self.df_records['power'].max()
                if pd.notna(max_power):
                    self.summary_data['max_power'] = str(int(max_power))
            
            if 'cadence_corrected' in self.df_records.columns:
                avg_cadence = self.df_records['cadence_corrected'].mean()
                if pd.notna(avg_cadence):
                    self.summary_data['avg_cadence'] = str(int(avg_cadence))
            
            if 'pace_min_km' in self.df_records.columns:
                avg_pace = self.df_records['pace_min_km'].mean()
                if pd.notna(avg_pace):
                    self.summary_data['avg_pace'] = self.safe_format_pace(avg_pace)
            
            if 'step_length_m' in self.df_records.columns:
                avg_step = self.df_records['step_length_m'].mean()
                if pd.notna(avg_step):
                    self.summary_data['avg_step_length'] = self.safe_format_float(avg_step, 2)
        
        # Elevación desde laps
        if not self.df_laps.empty:
            if 'total_ascent' in self.df_laps.columns:
                total_ascent = self.df_laps['total_ascent'].sum()
                self.summary_data['elevation_gain'] = str(int(total_ascent))
            
            if 'total_descent' in self.df_laps.columns:
                total_descent = self.df_laps['total_descent'].sum()
                self.summary_data['elevation_loss'] = str(int(total_descent))
    
    def _calculate_dynamics_data(self):
        """Calcula datos de dinámica de carrera"""
        self.dynamics_data = {
            'gct_left': 'N/A',
            'gct_right': 'N/A',
            'gct_imbalance': 'N/A',
            'vertical_oscillation': 'N/A',
            'step_length': 'N/A'
        }
        
        if self.df_records.empty:
            return
        
        if 'gct_left_ms' in self.df_records.columns:
            gct_left = self.df_records['gct_left_ms'].mean()
            if pd.notna(gct_left):
                self.dynamics_data['gct_left'] = str(int(gct_left))
        
        if 'gct_right_ms' in self.df_records.columns:
            gct_right = self.df_records['gct_right_ms'].mean()
            if pd.notna(gct_right):
                self.dynamics_data['gct_right'] = str(int(gct_right))
        
        if 'gct_imbalance_ms' in self.df_records.columns:
            imbalance = self.df_records['gct_imbalance_ms'].mean()
            if pd.notna(imbalance):
                self.dynamics_data['gct_imbalance'] = self.safe_format_float(imbalance, 1)
        
        if 'vertical_oscillation' in self.df_records.columns:
            vo = self.df_records['vertical_oscillation'].mean()
            if pd.notna(vo):
                self.dynamics_data['vertical_oscillation'] = str(int(vo))
        
        if 'step_length_m' in self.df_records.columns:
            step_len = self.df_records['step_length_m'].mean()
            if pd.notna(step_len):
                self.dynamics_data['step_length'] = self.safe_format_float(step_len, 2)
    
    def get_top_laps(self, n=3):
        """Obtiene los N mejores laps por pace"""
        if self.df_laps.empty or 'pace_calculated' not in self.df_laps.columns:
            return []
        
        valid_laps = self.df_laps.dropna(subset=['pace_calculated'])
        if valid_laps.empty:
            return []
        
        best_laps = valid_laps.nsmallest(n, 'pace_calculated')
        
        top_laps = []
        for idx, (lap_idx, lap) in enumerate(best_laps.iterrows(), 1):
            pace_str = self.safe_format_pace(lap['pace_calculated'])
            hr = int(lap['avg_heart_rate']) if pd.notna(lap.get('avg_heart_rate')) else 'N/A'
            power = int(lap['avg_power']) if pd.notna(lap.get('avg_power')) else 'N/A'
            
            # Distancia del lap
            distance = lap.get('total_distance', 0) / 1000
            distance_str = f"{distance:.1f}km"
            
            # Tiempo del lap
            time_str = self.safe_format_time(lap.get('total_timer_time', 0))
            
            # Cambio de altitud (ascent - descent)
            ascent = lap.get('total_ascent', 0)
            descent = lap.get('total_descent', 0)
            altitude_change = int(ascent - descent)
            altitude_str = f"{altitude_change:+d}m" if altitude_change != 0 else "±0m"
            
            top_laps.append({
                'rank': idx,
                'lap_number': lap_idx + 1,
                'pace': pace_str,
                'hr': hr,
                'power': power,
                'distance': distance_str,
                'time': time_str,
                'altitude_change': altitude_str
            })
        
        return top_laps
    
    def generate_lap_table_rows(self):
        """Genera las filas de la tabla de laps"""
        if self.df_laps.empty:
            return ""
        
        rows = []
        for i, (_, lap) in enumerate(self.df_laps.iterrows(), 1):
            # Inicializar todas las variables con valores por defecto
            row_data = {
                'lap_num': f"{i:02d}",
                'time_str': 'N/A',
                'distance_str': 'N/A',
                'cum_time_str': 'N/A',
                'cum_dist_str': 'N/A',
                'pace_str': 'N/A',
                'hr_avg': 'N/A',
                'hr_max': 'N/A',
                'power_avg': 'N/A',
                'power_max': 'N/A',
                'cadence_str': 'N/A',
                'step_str': 'N/A',
                'gct_str': 'N/A',
                'vert_str': 'N/A',
                'elev_str': '+0/-0'
            }
            
            # Procesar datos básicos
            if pd.notna(lap.get('total_timer_time')):
                row_data['time_str'] = self.safe_format_time(lap['total_timer_time'])
            
            if pd.notna(lap.get('total_distance')):
                distance = lap['total_distance'] / 1000
                row_data['distance_str'] = f"{distance:.1f}km"
            
            if pd.notna(lap.get('cumulative_time')):
                row_data['cum_time_str'] = self.safe_format_time(lap['cumulative_time'])
            
            if pd.notna(lap.get('cumulative_distance')):
                cum_dist = lap['cumulative_distance'] / 1000
                row_data['cum_dist_str'] = f"{cum_dist:.1f}km"
            
            if pd.notna(lap.get('pace_calculated')):
                row_data['pace_str'] = self.safe_format_pace(lap['pace_calculated'])
            
            # Heart Rate
            if pd.notna(lap.get('avg_heart_rate')):
                row_data['hr_avg'] = str(int(lap['avg_heart_rate']))
            if pd.notna(lap.get('max_heart_rate')):
                row_data['hr_max'] = str(int(lap['max_heart_rate']))
            
            # Power
            if pd.notna(lap.get('avg_power')):
                row_data['power_avg'] = str(int(lap['avg_power']))
            if pd.notna(lap.get('max_power')):
                row_data['power_max'] = str(int(lap['max_power']))
            
            # Cadencia
            if pd.notna(lap.get('avg_running_cadence_corrected')):
                row_data['cadence_str'] = str(int(lap['avg_running_cadence_corrected']))
            
            # Step Length
            if pd.notna(lap.get('avg_step_length_m')):
                row_data['step_str'] = self.safe_format_float(lap['avg_step_length_m'], 2)
            
            # Elevación
            ascent = int(lap.get('total_ascent', 0))
            descent = int(lap.get('total_descent', 0))
            row_data['elev_str'] = f"+{ascent}/-{descent}"
            
            # Generar fila HTML
            row_html = f"""
                        <tr>
                            <td>{row_data['lap_num']}</td>
                            <td>{row_data['time_str']}</td>
                            <td>{row_data['distance_str']}</td>
                            <td>{row_data['cum_time_str']}</td>
                            <td>{row_data['cum_dist_str']}</td>
                            <td>{row_data['pace_str']}</td>
                            <td>{row_data['hr_avg']}</td>
                            <td>{row_data['hr_max']}</td>
                            <td>{row_data['power_avg']}</td>
                            <td>{row_data['power_max']}</td>
                            <td>{row_data['cadence_str']}</td>
                            <td>{row_data['step_str']}</td>
                            <td>{row_data['gct_str']}</td>
                            <td>{row_data['vert_str']}</td>
                            <td>{row_data['elev_str']}</td>
                        </tr>"""
            
            rows.append(row_html)
        
        return '\n'.join(rows)
    
    def generate_pauses_html(self):
        """Genera HTML para la sección de pausas"""
        if not self.pauses:
            return """
                    <div class="pauses-left">
                        <div class="pause-list">
                            <div class="pause-item">
                                <div>
                                    <div class="pause-distance">✅ Sesión continua</div>
                                    <div class="pause-type">Sin pausas detectadas</div>
                                </div>
                                <div class="pause-duration">00:00</div>
                            </div>
                        </div>
                    </div>"""
        
        pause_items = []
        for pause in self.pauses:
            distance = pause['distance_km']
            duration = pause['duration_formatted']
            pause_type = pause['type']
            
            pause_items.append(f"""
                            <div class="pause-item">
                                <div>
                                    <div class="pause-distance">Km {distance:.1f}</div>
                                    <div class="pause-type">{pause_type}</div>
                                </div>
                                <div class="pause-duration">{duration}</div>
                            </div>""")
        
        return f"""
                    <div class="pauses-left">
                        <div class="pause-list">
                            {''.join(pause_items)}
                        </div>
                    </div>"""
    
    def generate_html_report(self, output_path=None):
        """Genera el reporte HTML completo"""
        # Información de sesión
        session_date = "N/A"
        if self.session_start_local:
            session_date = self.session_start_local.strftime('%d.%m.%Y • %H:%M')
        elif self.sessions and 'start_time' in self.sessions[0]:
            session_date = self.sessions[0]['start_time'].strftime('%d.%m.%Y • %H:%M UTC')
        
        location_display = self.location_info if self.location_info else "Ubicación no detectada"
        
        # Top laps
        top_laps = self.get_top_laps(3)
        top_laps_html = ""
        for lap_data in top_laps:
            top_laps_html += f"""
                <div class="top-lap">
                    <div class="lap-rank">{lap_data['rank']}</div>
                    <div class="lap-details">
                        <div class="lap-pace">{lap_data['pace']}/km</div>
                        <div class="lap-info">Lap {lap_data['lap_number']} • {lap_data['hr']} bpm • {lap_data['power']}W • {lap_data['time']} • {lap_data['distance']} • {lap_data['altitude_change']}</div>
                    </div>
                </div>"""
        
        # Filas de tabla de laps
        lap_table_rows = self.generate_lap_table_rows()
        
        # Pausas
        pauses_html = self.generate_pauses_html()
        
        # Template HTML completo
        html_template = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dream Report - Clean Analytics</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'IBM Plex Sans', sans-serif;
            background: #fafafa;
            color: #333;
            line-height: 1.6;
            padding: 40px 20px;
        }}
        
        .container {{
            max-width: 1000px;
            margin: 0 auto;
            background: white;
            box-shadow: 0 0 40px rgba(0,0,0,0.05);
            border: 1px solid #e5e5e5;
        }}
        
        .header {{
            padding: 60px 60px 40px;
            border-bottom: 2px solid #000;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            font-weight: 300;
            letter-spacing: -1px;
            margin-bottom: 20px;
        }}
        
        .session-info {{
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.95em;
            color: #666;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }}
        
        .info-item {{
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #eee;
        }}
        
        .label {{
            color: #999;
            font-weight: 400;
        }}
        
        .value {{
            font-weight: 500;
            color: #333;
        }}
        
        .metrics-section {{
            padding: 60px;
        }}
        
        .section-title {{
            font-size: 1.8em;
            font-weight: 300;
            margin-bottom: 40px;
            letter-spacing: -0.5px;
        }}
        
        .subsection-title {{
            font-size: 1.2em;
            font-weight: 400;
            margin-bottom: 25px;
            margin-top: 40px;
            color: #333;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-size: 0.9em;
        }}
        
        .subsection-title:first-child {{
            margin-top: 0;
        }}
        
        .metrics-subsection {{
            margin-bottom: 50px;
        }}
        
        .metrics-subsection:last-child {{
            margin-bottom: 0;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 40px;
            margin-bottom: 30px;
        }}
        
        .metric {{
            text-align: center;
        }}
        
        .metric-value {{
            font-family: 'IBM Plex Mono', monospace;
            font-size: 3em;
            font-weight: 500;
            color: #000;
            line-height: 1;
            margin-bottom: 8px;
        }}
        
        .metric-label {{
            font-size: 0.85em;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .metric-unit {{
            font-size: 0.7em;
            color: #999;
            margin-left: 4px;
        }}
        
        .divider {{
            height: 1px;
            background: #eee;
            margin: 60px 0;
        }}
        
        .dynamics-section {{
            padding: 0 60px 60px;
        }}
        
        .dynamics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 30px;
            text-align: center;
        }}
        
        .dynamic-item {{
            padding: 20px 0;
        }}
        
        .dynamic-value {{
            font-family: 'IBM Plex Mono', monospace;
            font-size: 1.8em;
            font-weight: 500;
            color: #000;
            margin-bottom: 8px;
        }}
        
        .dynamic-label {{
            font-size: 0.8em;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .table-section {{
            padding: 0 60px 60px;
        }}
        
        .lap-table {{
            width: 100%;
            border-collapse: collapse;
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.85em;
            margin-top: 30px;
        }}
        
        .lap-table th {{
            background: #f8f8f8;
            padding: 12px 8px;
            text-align: center;
            font-weight: 500;
            color: #333;
            border-bottom: 2px solid #ddd;
            font-size: 0.75em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .lap-table td {{
            padding: 10px 8px;
            text-align: center;
            border-bottom: 1px solid #eee;
            color: #333;
        }}
        
        .lap-table tbody tr:nth-child(odd) {{
            background: #fafafa;
        }}
        
        .lap-table tbody tr:hover {{
            background: #f0f0f0;
        }}
        
        .pauses-section {{
            padding: 0 60px 60px;
        }}
        
        .pauses-container {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 40px;
            margin-top: 30px;
        }}
        
        .pauses-right {{
            background: #fafafa;
            border: 1px dashed #ddd;
            min-height: 200px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #999;
            font-style: italic;
        }}
        
        .pause-item {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 16px 0;
            border-bottom: 1px solid #eee;
        }}
        
        .pause-distance {{
            font-family: 'IBM Plex Mono', monospace;
            font-weight: 500;
        }}
        
        .pause-duration {{
            font-family: 'IBM Plex Mono', monospace;
            color: #666;
        }}
        
        .pause-type {{
            font-size: 0.85em;
            color: #999;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .footer {{
            background: #000;
            color: white;
            text-align: center;
            padding: 30px;
            font-size: 0.9em;
        }}
        
        .top-laps {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }}
        
        .top-lap {{
            display: flex;
            align-items: center;
            padding: 20px;
            border: 1px solid #eee;
            border-radius: 0;
        }}
        
        .lap-rank {{
            font-family: 'IBM Plex Mono', monospace;
            font-size: 2em;
            font-weight: 600;
            color: #000;
            margin-right: 20px;
            min-width: 40px;
        }}
        
        .lap-pace {{
            font-family: 'IBM Plex Mono', monospace;
            font-size: 1.5em;
            font-weight: 600;
            color: #000;
            margin-bottom: 5px;
        }}
        
        .lap-info {{
            font-size: 0.9em;
            color: #666;
        }}
        
        @media (max-width: 768px) {{
            .header,
            .metrics-section,
            .dynamics-section,
            .table-section,
            .pauses-section {{
                padding-left: 30px;
                padding-right: 30px;
            }}
            
            .metrics-grid {{
                grid-template-columns: repeat(2, 1fr);
                gap: 30px;
            }}
            
            .dynamics-grid {{
                grid-template-columns: repeat(2, 1fr);
            }}
            
            .lap-table {{
                font-size: 0.75em;
            }}
            
            .lap-table th,
            .lap-table td {{
                padding: 8px 4px;
            }}
            
            .pauses-container {{
                grid-template-columns: 1fr;
                gap: 20px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>Dream Report</h1>
            <div class="session-info">
                <div class="info-item">
                    <span class="label">Fecha</span>
                    <span class="value">{session_date}</span>
                </div>
                <div class="info-item">
                    <span class="label">Ubicación</span>
                    <span class="value">{location_display}</span>
                </div>
                <div class="info-item">
                    <span class="label">Deporte</span>
                    <span class="value">Running</span>
                </div>
                <div class="info-item">
                    <span class="label">Laps</span>
                    <span class="value">{self.summary_data['total_laps']} laps</span>
                </div>
                <div class="info-item">
                    <span class="label">Elevación</span>
                    <span class="value">+{self.summary_data['elevation_gain']}m / -{self.summary_data['elevation_loss']}m</span>
                </div>
            </div>
        </div>
        
        <!-- Main Metrics -->
        <div class="metrics-section">
            <h2 class="section-title">Métricas Principales</h2>
            
            <!-- Datos de Sesión -->
            <div class="metrics-subsection">
                <h3 class="subsection-title">Datos de Sesión</h3>
                <div class="metrics-grid">
                    <div class="metric">
                        <div class="metric-value">{self.summary_data['duration']}<span class="metric-unit">min</span></div>
                        <div class="metric-label">Duración</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{self.summary_data['distance']}<span class="metric-unit">km</span></div>
                        <div class="metric-label">Distancia</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{self.summary_data['avg_pace']}<span class="metric-unit">/km</span></div>
                        <div class="metric-label">Pace Promedio</div>
                    </div>
                </div>
            </div>
            
            <!-- Esfuerzo Físico -->
            <div class="metrics-subsection">
                <h3 class="subsection-title">Esfuerzo Físico</h3>
                <div class="metrics-grid">
                    <div class="metric">
                        <div class="metric-value">{self.summary_data['avg_hr']}<span class="metric-unit">bpm</span></div>
                        <div class="metric-label">HR Promedio</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{self.summary_data['max_hr']}<span class="metric-unit">bpm</span></div>
                        <div class="metric-label">HR Máximo</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{self.summary_data['calories']}<span class="metric-unit">kcal</span></div>
                        <div class="metric-label">Calorías</div>
                    </div>
                </div>
            </div>
            
            <!-- Technical Running -->
            <div class="metrics-subsection">
                <h3 class="subsection-title">Technical Running</h3>
                <div class="metrics-grid">
                    <div class="metric">
                        <div class="metric-value">{self.summary_data['avg_cadence']}<span class="metric-unit">spm</span></div>
                        <div class="metric-label">Cadencia</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{self.summary_data['avg_step_length']}<span class="metric-unit">m</span></div>
                        <div class="metric-label">Step Length</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{self.summary_data['avg_power']}<span class="metric-unit">W</span></div>
                        <div class="metric-label">Potencia Avg</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{self.summary_data['max_power']}<span class="metric-unit">W</span></div>
                        <div class="metric-label">Potencia Max</div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="divider"></div>
        
        <!-- Top 3 Mejores Laps -->
        <div class="table-section">
            <h2 class="section-title">Top 3 Laps Más Rápidos</h2>
            <div class="top-laps">
                {top_laps_html}
            </div>
        </div>
        
        <!-- Running Dynamics -->
        <div class="dynamics-section">
            <h2 class="section-title">Dinámica de Carrera</h2>
            <div class="dynamics-grid">
                <div class="dynamic-item">
                    <div class="dynamic-value">{self.dynamics_data['gct_left']}</div>
                    <div class="dynamic-label">GCT Izq (ms)</div>
                </div>
                <div class="dynamic-item">
                    <div class="dynamic-value">{self.dynamics_data['gct_right']}</div>
                    <div class="dynamic-label">GCT Der (ms)</div>
                </div>
                <div class="dynamic-item">
                    <div class="dynamic-value">{self.dynamics_data['gct_imbalance']}</div>
                    <div class="dynamic-label">Desbalance (ms)</div>
                </div>
                <div class="dynamic-item">
                    <div class="dynamic-value">{self.dynamics_data['vertical_oscillation']}</div>
                    <div class="dynamic-label">Osc Vert (mm)</div>
                </div>
                <div class="dynamic-item">
                    <div class="dynamic-value">{self.dynamics_data['step_length']}</div>
                    <div class="dynamic-label">Step Len (m)</div>
                </div>
            </div>
        </div>
        
        <div class="divider"></div>
        
        <!-- Lap Analysis -->
        <div class="table-section">
            <h2 class="section-title">Análisis por Lap</h2>
            <table class="lap-table">
                <thead>
                    <tr>
                        <th>Lap</th>
                        <th>Tiempo</th>
                        <th>Dist</th>
                        <th>Cum Time</th>
                        <th>Cum Dist</th>
                        <th>Pace</th>
                        <th>HR Avg</th>
                        <th>HR Max</th>
                        <th>Power Avg</th>
                        <th>Power Max</th>
                        <th>Cadence</th>
                        <th>Step Len</th>
                        <th>GCT Imb</th>
                        <th>Vert Ratio</th>
                        <th>Elev</th>
                    </tr>
                </thead>
                <tbody>
                    {lap_table_rows}
                </tbody>
            </table>
        </div>
        
        <div class="divider"></div>
        
        <!-- Pauses -->
        <div class="pauses-section">
            <h2 class="section-title">Pausas</h2>
            <div class="pauses-container">
                {pauses_html}
                <div class="pauses-right">
                    Generado automáticamente desde archivo FIT
                </div>
            </div>
        </div>
        
        <!-- Footer -->
        <div class="footer">
            Dream Report • Análisis de Rendimiento para Atletas
        </div>
    </div>
</body>
</html>"""
        
        # Guardar archivo HTML
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dream_report_{timestamp}.html"
        else:
            filename = output_path
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(html_template)
            print(f"✅ Reporte HTML generado: {filename}")
            return filename
        except Exception as e:
            self.errors.append(f"Error guardando HTML: {e}")
            return None

def main():
    """Función principal del script"""
    if len(sys.argv) != 2:
        print("Uso: python fit_to_html.py archivo.fit")
        print("\nEjemplo:")
        print("  python fit_to_html.py mi_carrera.fit")
        print("\nEsto generará un archivo HTML con el nombre dream_report_YYYYMMDD_HHMMSS.html")
        sys.exit(1)
    
    fit_file = sys.argv[1]
    
    if not os.path.exists(fit_file):
        print(f"❌ Archivo no encontrado: {fit_file}")
        sys.exit(1)
    
    # Validar extensión
    if not fit_file.lower().endswith('.fit'):
        print(f"⚠️  Advertencia: El archivo no tiene extensión .fit: {fit_file}")
        response = input("¿Continuar de todas formas? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    print("🏃‍♂️ Iniciando generación de Dream Report HTML")
    print("=" * 60)
    
    # Crear y ejecutar generador
    generator = FitToHtmlGenerator(fit_file)
    
    # Cargar datos
    if not generator.load_fit_file():
        print("❌ Error cargando archivo FIT")
        if generator.errors:
            for error in generator.errors:
                print(f"   {error}")
        sys.exit(1)
    
    # Procesar datos
    print("📊 Procesando datos...")
    generator.process_data()
    
    # Generar HTML
    print("🎨 Generando reporte HTML...")
    html_file = generator.generate_html_report()
    
    # Reporte final
    print("\n" + "=" * 60)
    if html_file:
        print(f"🎉 REPORTE GENERADO EXITOSAMENTE")
        print(f"📄 Archivo: {html_file}")
        print(f"📊 Datos procesados:")
        print(f"   • {len(generator.records)} records")
        print(f"   • {len(generator.laps)} laps válidos")
        print(f"   • {len(generator.pauses)} pausas detectadas")
        
        if generator.location_info:
            print(f"   • Ubicación: {generator.location_info}")
        
        if generator.errors:
            print(f"\n⚠️  {len(generator.errors)} errores durante procesamiento:")
            for error in generator.errors[:3]:  # Mostrar solo los primeros 3
                print(f"   • {error}")
        
        if generator.warnings:
            print(f"\n💡 {len(generator.warnings)} advertencias:")
            for warning in generator.warnings[:3]:  # Mostrar solo las primeras 3
                print(f"   • {warning}")
        
        print(f"\n🌐 Abre el archivo en tu navegador para ver el reporte")
    else:
        print("❌ ERROR: No se pudo generar el reporte HTML")
        if generator.errors:
            for error in generator.errors:
                print(f"   {error}")
        sys.exit(1)
    
    # Tips de instalación si faltan librerías opcionales
    if not TIMEZONE_FINDER_AVAILABLE:
        print("\n💡 TIP: Instala 'timezonefinder' para detección automática de zona horaria:")
        print("   pip install timezonefinder")
    
    if not REVERSE_GEOCODER_AVAILABLE:
        print("\n💡 TIP: Instala 'reverse_geocoder' para detección de ubicación:")
        print("   pip install reverse_geocoder")

if __name__ == "__main__":
    main()
