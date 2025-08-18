#!/usr/bin/env python3
"""
Dream Report - Análisis profesional de archivos FIT para running de alto nivel
Versión simplificada usando local_timestamp directo del archivo FIT
"""

import fitdecode
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
from pathlib import Path

# Opcional: para detección de ubicación desde GPS (solo para mostrar ubicación)
try:
    import reverse_geocoder as rg
    REVERSE_GEOCODER_AVAILABLE = True
except ImportError:
    REVERSE_GEOCODER_AVAILABLE = False

class FitDreamAnalyzer:
    def __init__(self, fit_file_path):
        self.fit_file_path = fit_file_path
        self.records = []
        self.laps = []
        self.sessions = []
        self.events = []
        self.device_info = []
        self.activities = []  # Agregar activities
        
        # DataFrames procesados
        self.df_records = None
        self.df_laps = None
        self.df_events = None
        
        # Análisis
        self.pauses = []
        self.key_devices = []
        self.session_start_local = None
        self.location_info = None
        self.timezone_offset = None
        
    def load_fit_file(self):
        """Carga y procesa el archivo FIT"""
        print(f"📂 Analizando: {os.path.basename(self.fit_file_path)}")
        
        try:
            with fitdecode.FitReader(self.fit_file_path) as fit:
                for frame in fit:
                    if isinstance(frame, fitdecode.FitDataMessage):
                        self._process_message(frame)
        except Exception as e:
            print(f"❌ Error: {e}")
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
        elif msg_type == 'activity':  # Agregar activity
            self.activities.append(data)
    
    def process_data(self):
        """Convierte y procesa todos los datos"""
        if self.records:
            self.df_records = pd.DataFrame(self.records)
        if self.laps:
            self.df_laps = pd.DataFrame(self.laps)
        if self.events:
            self.df_events = pd.DataFrame(self.events)
        
        self._process_timestamps()
        self._extract_local_timestamp()
        self._detect_location()
        self._calculate_derived_metrics()
        self._detect_pauses()
        self._identify_key_devices()
        self._calculate_lap_metrics()
        self._filter_invalid_laps()
        self._calculate_cumulative_metrics()
    
    def _extract_local_timestamp(self):
        """Extrae local_timestamp desde activity o calcula desde nombre del archivo"""
        local_found = False
        
        # 1. Buscar en activities primero
        if self.activities and 'local_timestamp' in self.activities[0]:
            activity_local = self.activities[0]['local_timestamp']
            activity_utc = self.activities[0].get('timestamp')
            
            if activity_local and activity_utc:
                # Normalizar ambos datetimes para poder compararlos
                if activity_local.tzinfo is not None and activity_utc.tzinfo is None:
                    # activity_local tiene timezone, activity_utc no
                    activity_utc = activity_utc.replace(tzinfo=activity_local.tzinfo)
                elif activity_local.tzinfo is None and activity_utc.tzinfo is not None:
                    # activity_utc tiene timezone, activity_local no
                    activity_local = activity_local.replace(tzinfo=activity_utc.tzinfo)
                
                self.session_start_local = activity_local
                offset_hours = (activity_local - activity_utc).total_seconds() / 3600
                self.timezone_offset = int(round(offset_hours))
                print(f"🕒 Hora local desde activity: {activity_local.strftime('%d/%m/%Y %H:%M:%S')} (UTC{self.timezone_offset:+d})")
                local_found = True
        
        # 2. Buscar en session como fallback
        if not local_found and self.sessions and 'local_timestamp' in self.sessions[0]:
            session_local = self.sessions[0]['local_timestamp']
            session_utc = self.sessions[0].get('start_time', self.sessions[0].get('timestamp'))
            
            if session_local and session_utc:
                # Normalizar ambos datetimes
                if session_local.tzinfo is not None and session_utc.tzinfo is None:
                    session_utc = session_utc.replace(tzinfo=session_local.tzinfo)
                elif session_local.tzinfo is None and session_utc.tzinfo is not None:
                    session_local = session_local.replace(tzinfo=session_utc.tzinfo)
                
                self.session_start_local = session_local
                offset_hours = (session_local - session_utc).total_seconds() / 3600
                self.timezone_offset = int(round(offset_hours))
                print(f"🕒 Hora local desde session: {session_local.strftime('%d/%m/%Y %H:%M:%S')} (UTC{self.timezone_offset:+d})")
                local_found = True
        
        # 3. Calcular desde nombre del archivo como último recurso
        if not local_found:
            filename = os.path.basename(self.fit_file_path)
            session_utc = None
            
            if self.sessions:
                session_utc = self.sessions[0].get('start_time', self.sessions[0].get('timestamp'))
            
            if session_utc and 'h' in filename:
                try:
                    # Buscar patrón como "17h27"
                    parts = filename.split('_')
                    for part in parts:
                        if 'h' in part:
                            time_part = part.replace('.fit', '')
                            hour, minute = time_part.split('h')
                            
                            # Crear timestamp local aproximado
                            local_hour = int(hour)
                            local_minute = int(minute)
                            
                            # Obtener hora UTC sin timezone para comparar
                            if session_utc.tzinfo is not None:
                                utc_hour = session_utc.hour
                                utc_minute = session_utc.minute
                            else:
                                utc_hour = session_utc.hour
                                utc_minute = session_utc.minute
                            
                            # Calcular offset
                            offset_hours = (local_hour - utc_hour) % 24
                            if offset_hours > 12:
                                offset_hours -= 24
                            
                            # Crear timestamp local estimado
                            from datetime import timedelta
                            if session_utc.tzinfo is not None:
                                # Remover timezone para hacer el cálculo
                                utc_naive = session_utc.replace(tzinfo=None)
                                self.session_start_local = utc_naive + timedelta(hours=offset_hours)
                            else:
                                self.session_start_local = session_utc + timedelta(hours=offset_hours)
                            
                            self.timezone_offset = offset_hours
                            
                            print(f"🕒 Hora calculada desde filename: {self.session_start_local.strftime('%d/%m/%Y %H:%M:%S')} (UTC{self.timezone_offset:+d})")
                            local_found = True
                            break
                except Exception as e:
                    print(f"⚠️  Error calculando desde filename: {e}")
        
        # 4. Fallback final a UTC
        if not local_found and self.sessions:
            session_utc = self.sessions[0].get('start_time', self.sessions[0].get('timestamp'))
            if session_utc:
                # Remover timezone info si existe para tratarlo como local
                if session_utc.tzinfo is not None:
                    self.session_start_local = session_utc.replace(tzinfo=None)
                else:
                    self.session_start_local = session_utc
                self.timezone_offset = 0
                print(f"🕒 Usando start_time (UTC): {self.session_start_local.strftime('%d/%m/%Y %H:%M:%S')}")
        
        if not local_found and not self.sessions:
            print("⚠️  No se encontró información de timestamp")
    
    def _detect_location(self):
        """Detecta ubicación desde coordenadas GPS (solo para mostrar ubicación)"""
        if not REVERSE_GEOCODER_AVAILABLE:
            return
        
        # Buscar coordenadas GPS en records
        if self.df_records is None:
            return
        
        if 'position_lat' in self.df_records.columns and 'position_long' in self.df_records.columns:
            # Obtener primera coordenada válida
            valid_coords = self.df_records[
                (self.df_records['position_lat'].notna()) & 
                (self.df_records['position_long'].notna()) &
                (self.df_records['position_lat'] != 0) &
                (self.df_records['position_long'] != 0)
            ]
            
            if len(valid_coords) > 0:
                # Las coordenadas en FIT vienen en semicircles, convertir a grados
                lat_semicircles = valid_coords['position_lat'].iloc[0]
                lng_semicircles = valid_coords['position_long'].iloc[0]
                
                # Conversión: grados = semicircles * (180 / 2^31)
                lat = lat_semicircles * (180 / (2**31))
                lng = lng_semicircles * (180 / (2**31))
                
                # Detectar ubicación con reverse geocoding
                try:
                    location_results = rg.search([(lat, lng)])
                    if location_results:
                        location = location_results[0]
                        city = location.get('name', 'Unknown')
                        admin1 = location.get('admin1', '')
                        country = location.get('cc', '')
                        
                        self.location_info = f"{city}, {admin1}, {country}"
                        print(f"🏃 Ubicación: {self.location_info}")
                except Exception as e:
                    print(f"⚠️  Error detectando ubicación: {e}")
    
    def _process_timestamps(self):
        """Procesa timestamps para análisis temporal"""
        if self.df_records is not None and 'timestamp' in self.df_records.columns:
            self.df_records['timestamp'] = pd.to_datetime(self.df_records['timestamp'])
            if len(self.df_records) > 1:
                self.df_records['elapsed_seconds'] = (
                    self.df_records['timestamp'] - self.df_records['timestamp'].iloc[0]
                ).dt.total_seconds()
    
    def _calculate_derived_metrics(self):
        """Calcula métricas derivadas esenciales"""
        if self.df_records is None:
            return
        
        # CORREGIR CADENCIA - multiplicar por 2
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
    
    def _calculate_lap_metrics(self):
        """Calcula métricas por lap desde records"""
        if self.df_laps is None or self.df_records is None:
            return
        
        # Calcular pace real por lap
        if 'total_timer_time' in self.df_laps.columns and 'total_distance' in self.df_laps.columns:
            self.df_laps['pace_calculated'] = self.df_laps.apply(
                lambda row: (row['total_timer_time'] / 60) / (row['total_distance'] / 1000) 
                if row['total_distance'] > 0 else np.nan, axis=1
            )
        
        # Corregir cadencia en laps - multiplicar por 2
        if 'avg_running_cadence' in self.df_laps.columns:
            self.df_laps['avg_running_cadence_corrected'] = self.df_laps['avg_running_cadence'] * 2
        if 'max_running_cadence' in self.df_laps.columns:
            self.df_laps['max_running_cadence_corrected'] = self.df_laps['max_running_cadence'] * 2
        
        # Step length en metros
        if 'avg_step_length' in self.df_laps.columns:
            self.df_laps['avg_step_length_m'] = self.df_laps['avg_step_length'] / 1000
        
        # Calcular GCT desbalance por lap desde records (aproximación)
        if 'gct_imbalance_ms' in self.df_records.columns and len(self.df_records) > 0:
            records_per_lap = len(self.df_records) // len(self.df_laps)
            gct_imbalance = []
            
            for i in range(len(self.df_laps)):
                start_idx = i * records_per_lap
                end_idx = min((i + 1) * records_per_lap, len(self.df_records))
                
                if start_idx < len(self.df_records):
                    lap_records = self.df_records.iloc[start_idx:end_idx]
                    avg_imbalance = lap_records['gct_imbalance_ms'].mean()
                    gct_imbalance.append(avg_imbalance)
                else:
                    gct_imbalance.append(np.nan)
            
            self.df_laps['gct_imbalance_ms'] = gct_imbalance
    
    def _filter_invalid_laps(self):
        """Filtra laps inválidos (duración menor a 10 segundos)"""
        if self.df_laps is None or len(self.df_laps) == 0:
            return
        
        original_count = len(self.df_laps)
        
        # Filtrar laps con duración menor a 10 segundos
        if 'total_timer_time' in self.df_laps.columns:
            valid_laps_mask = self.df_laps['total_timer_time'] >= 10
            invalid_laps = self.df_laps[~valid_laps_mask]
            
            if len(invalid_laps) > 0:
                print(f"🚫 Filtrando {len(invalid_laps)} lap(s) inválido(s) (< 10 segundos):")
                for idx, lap in invalid_laps.iterrows():
                    duration = lap['total_timer_time']
                    lap_num = idx + 1
                    print(f"   - Lap {lap_num}: {duration:.1f}s (descartado)")
            
            # Aplicar filtro
            self.df_laps = self.df_laps[valid_laps_mask].reset_index(drop=True)
            
            # Actualizar también la lista original de laps
            self.laps = [lap for i, lap in enumerate(self.laps) if i < len(valid_laps_mask) and valid_laps_mask.iloc[i]]
            
            filtered_count = len(self.df_laps)
            if original_count != filtered_count:
                print(f"✅ Laps válidos para análisis: {filtered_count} de {original_count}")
    
    def _calculate_cumulative_metrics(self):
        """Calcula métricas acumuladas por lap"""
        if self.df_laps is None:
            return
        
        # Tiempo acumulado
        if 'total_timer_time' in self.df_laps.columns:
            self.df_laps['cumulative_time'] = self.df_laps['total_timer_time'].cumsum()
        
        # Distancia acumulada
        if 'total_distance' in self.df_laps.columns:
            self.df_laps['cumulative_distance'] = self.df_laps['total_distance'].cumsum()
    
    def _format_time(self, seconds):
        """Formatea segundos a formato HH:MM:SS o MM:SS"""
        if pd.isna(seconds):
            return "N/A"
        
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes}:{secs:02d}"
    
    def _detect_pauses(self):
        """Detecta pausas con distancia y duración"""
        if self.df_events is None or self.df_records is None:
            return
        
        timer_events = self.df_events[
            (self.df_events['event'] == 'timer') & 
            (self.df_events['event_type'].isin(['start', 'stop_all']))
        ].copy()
        
        if len(timer_events) < 2:
            return
        
        timer_events = timer_events.sort_values('timestamp').reset_index(drop=True)
        
        pauses = []
        for i in range(len(timer_events) - 1):
            if (timer_events.iloc[i]['event_type'] == 'stop_all' and 
                timer_events.iloc[i + 1]['event_type'] == 'start'):
                
                stop_time = timer_events.iloc[i]['timestamp']
                start_time = timer_events.iloc[i + 1]['timestamp']
                duration = (start_time - stop_time).total_seconds()
                
                # Encontrar distancia al momento de la pausa
                pause_record = self.df_records[self.df_records['timestamp'] <= stop_time]
                if not pause_record.empty and 'distance' in pause_record.columns:
                    distance_km = pause_record['distance'].iloc[-1] / 1000
                    pauses.append({
                        'distance_km': distance_km,
                        'duration_seconds': duration,
                        'duration_formatted': str(timedelta(seconds=int(duration)))
                    })
        
        self.pauses = pauses
    
    def _identify_key_devices(self):
        """Identifica dispositivos clave únicos"""
        unique_devices = []
        seen_serials = set()
        
        for device in self.device_info:
            serial = device.get('serial_number')
            device_type = device.get('antplus_device_type', device.get('device_type'))
            
            if serial and serial not in seen_serials:
                seen_serials.add(serial)
                device_info = {
                    'serial': serial,
                    'type': device_type,
                    'manufacturer': device.get('manufacturer', 'unknown'),
                    'product': device.get('garmin_product', device.get('product', 'unknown')),
                    'battery_status': device.get('battery_status'),
                    'battery_voltage': device.get('battery_voltage')
                }
                unique_devices.append(device_info)
        
        self.key_devices = unique_devices
    
    def generate_report(self):
        """Genera el Dream Report profesional"""
        self._print_header()
        self._print_executive_summary()
        self._print_running_dynamics_summary()
        self._print_detailed_lap_analysis()
        self._print_pauses_analysis()
        self._print_equipment_summary()
        self._print_footer()
    
    def _print_header(self):
        """Header del reporte"""
        print("\n" + "="*80)
        print("🏃 DREAM REPORT - ANÁLISIS DE RENDIMIENTO")
        print("="*80)
        
        if self.session_start_local:
            if self.timezone_offset is not None and self.timezone_offset != 0:
                date_str = self.session_start_local.strftime('%d/%m/%Y %H:%M') + f" (UTC{self.timezone_offset:+d})"
            else:
                date_str = self.session_start_local.strftime('%d/%m/%Y %H:%M (hora local)')
            print(f"📅 Sesión: {date_str}")
        
        # Mostrar ubicación si está disponible
        if self.location_info:
            print(f"📍 Ubicación: {self.location_info}")
    
    def _print_executive_summary(self):
        """1. Resumen Ejecutivo"""
        print(f"\n🎯 RESUMEN EJECUTIVO")
        print("-" * 40)
        
        if self.sessions:
            session = self.sessions[0]
            
            # Métricas principales
            if 'total_elapsed_time' in session:
                duration = str(timedelta(seconds=int(session['total_elapsed_time'])))
                print(f"⏱️  Duración: {duration}")
            
            if 'total_distance' in session:
                distance = session['total_distance'] / 1000
                print(f"📏 Distancia: {distance:.2f} km")
            
            # Pace promedio de la sesión
            if 'total_timer_time' in session and 'total_distance' in session:
                timer_time_minutes = session['total_timer_time'] / 60
                distance_km = session['total_distance'] / 1000
                if distance_km > 0:
                    avg_pace = timer_time_minutes / distance_km
                    pace_min = int(avg_pace)
                    pace_sec = int((avg_pace % 1) * 60)
                    print(f"🏃‍♂️ Pace Promedio: {pace_min}:{pace_sec:02d}/km")
            
            # Heart Rate de la sesión
            if 'avg_heart_rate' in session and 'max_heart_rate' in session:
                avg_hr = int(session['avg_heart_rate'])
                max_hr = int(session['max_heart_rate'])
                print(f"💗 Frecuencia Cardíaca: {avg_hr} bpm promedio | {max_hr} bpm máximo")
            elif 'avg_heart_rate' in session:
                avg_hr = int(session['avg_heart_rate'])
                print(f"💗 Frecuencia Cardíaca: {avg_hr} bpm promedio")
            elif 'max_heart_rate' in session:
                max_hr = int(session['max_heart_rate'])
                print(f"💗 Frecuencia Cardíaca: {max_hr} bpm máximo")
            
            if 'normalized_power' in session:
                print(f"💪 Potencia Normalizada: {session['normalized_power']} W")
            
            # Calorías - intentar distinguir entre activas y totales
            calories_info = self._get_calories_info(session)
            if calories_info:
                print(calories_info)
            
            # TSS si existe
            if 'training_stress_score' in session:
                print(f"📊 TSS: {session['training_stress_score']}")
        
        # Elevación
        if self.df_laps is not None and 'total_ascent' in self.df_laps.columns:
            ascent = self.df_laps['total_ascent'].sum()
            descent = self.df_laps.get('total_descent', pd.Series([0])).sum()
            print(f"⛰️  Elevación: +{ascent}m / -{descent}m")
        
        print(f"🔄 Vueltas válidas: {len(self.laps)}")
        print(f"⏸️  Pausas: {len(self.pauses)}")
    
    def _get_calories_info(self, session):
        """Determina qué tipo de calorías están disponibles y las formatea correctamente"""
        # Verificar si tenemos calorías por lap para hacer cálculos más precisos
        has_lap_calories = (self.df_laps is not None and 
                           'total_calories' in self.df_laps.columns and 
                           not self.df_laps['total_calories'].isna().all())
        
        if has_lap_calories:
            # Sumar calorías de todos los laps válidos
            lap_calories_sum = self.df_laps['total_calories'].sum()
            session_calories = session.get('total_calories', 0)
            
            # Si hay diferencia significativa, mostrar ambos
            if abs(lap_calories_sum - session_calories) > 5:
                return f"🔥 Calorías: {int(lap_calories_sum)} (por laps) | {int(session_calories)} (sesión total)"
            else:
                # Si son similares, usar el de laps que es más preciso
                return f"🔥 Calorías Totales: {int(lap_calories_sum)}"
        
        # Fallback al método original si no hay datos de laps
        if 'total_calories' in session:
            return f"🔥 Calorías Totales: {session['total_calories']}"
        
        return None
    
    def _print_running_dynamics_summary(self):
        """2. Resumen de Dinámica de Carrera Global"""
        print(f"\n🦶 DINÁMICA DE CARRERA - PROMEDIOS GLOBALES")
        print("-" * 50)
        
        if self.df_records is None:
            print("Sin datos de records disponibles")
            return
        
        # Ground Contact Time global
        if 'gct_left_ms' in self.df_records.columns:
            gct_left = self.df_records['gct_left_ms'].mean()
            gct_right = self.df_records['gct_right_ms'].mean()
            imbalance = abs(gct_left - gct_right)
            
            print(f"👟 Contacto con suelo: {gct_left:.0f}ms (izq) | {gct_right:.0f}ms (der)")
            print(f"⚖️  Desbalance promedio: {imbalance:.1f}ms")
        
        # Otras métricas globales
        if 'vertical_oscillation' in self.df_records.columns:
            vo_avg = self.df_records['vertical_oscillation'].mean()
            print(f"📊 Oscilación Vertical: {vo_avg:.0f} mm promedio")
        
        if 'stance_time' in self.df_records.columns:
            st_avg = self.df_records['stance_time'].mean()
            print(f"⏱️  Tiempo de Contacto: {st_avg:.0f} ms promedio")
        
        if 'cadence_corrected' in self.df_records.columns:
            cadence_avg = self.df_records['cadence_corrected'].mean()
            print(f"🎯 Cadencia Global: {cadence_avg:.0f} spm promedio")
    
    def _print_detailed_lap_analysis(self):
        """3. Análisis Detallado por Lap con métricas acumuladas"""
        print(f"\n🏁 ANÁLISIS DETALLADO POR LAP")
        print("-" * 100)
        
        if self.df_laps is None or len(self.df_laps) == 0:
            print("No hay datos de laps disponibles")
            return
        
        # Header de la tabla (expandido)
        header = "Lap | Pace    | Dist  | Cum Time | Cum Dist | HR Avg/Max | Power Avg/Max | Cadence Avg/Max | Step Length | GCT Imb | Elev +/-"
        print(header)
        print("-" * len(header))
        
        for i, (_, lap) in enumerate(self.df_laps.iterrows(), 1):
            # Pace
            if pd.notna(lap.get('pace_calculated')):
                pace_min = int(lap['pace_calculated'])
                pace_sec = int((lap['pace_calculated'] % 1) * 60)
                pace_str = f"{pace_min}:{pace_sec:02d}"
            else:
                pace_str = "N/A"
            
            # Distancia
            distance = lap.get('total_distance', 0) / 1000
            dist_str = f"{distance:.1f}km"
            
            # Tiempo acumulado
            cum_time = lap.get('cumulative_time', 0)
            cum_time_str = self._format_time(cum_time)
            
            # Distancia acumulada
            cum_dist = lap.get('cumulative_distance', 0) / 1000
            cum_dist_str = f"{cum_dist:.1f}km"
            
            # Heart Rate
            hr_avg = int(lap['avg_heart_rate']) if pd.notna(lap.get('avg_heart_rate')) else 'N/A'
            hr_max = int(lap['max_heart_rate']) if pd.notna(lap.get('max_heart_rate')) else 'N/A'
            hr_str = f"{hr_avg}/{hr_max}"
            
            # Power
            power_avg = int(lap['avg_power']) if pd.notna(lap.get('avg_power')) else 'N/A'
            power_max = int(lap['max_power']) if pd.notna(lap.get('max_power')) else 'N/A'
            power_str = f"{power_avg}/{power_max}"
            
            # Cadencia corregida
            cadence_avg = int(lap['avg_running_cadence_corrected']) if pd.notna(lap.get('avg_running_cadence_corrected')) else 'N/A'
            cadence_max = int(lap['max_running_cadence_corrected']) if pd.notna(lap.get('max_running_cadence_corrected')) else 'N/A'
            cadence_str = f"{cadence_avg}/{cadence_max}"
            
            # Step Length en metros
            step_length = lap.get('avg_step_length_m', 0)
            step_str = f"{step_length:.2f}m" if step_length > 0 else 'N/A'
            
            # GCT Imbalance
            gct_imb = lap.get('gct_imbalance_ms')
            gct_str = f"{gct_imb:.1f}ms" if pd.notna(gct_imb) else 'N/A'
            
            # Elevación
            ascent = int(lap.get('total_ascent', 0))
            descent = int(lap.get('total_descent', 0))
            elev_str = f"+{ascent}/-{descent}"
            
            # Formatear fila
            row = f"{i:2d}  | {pace_str:7s} | {dist_str:5s} | {cum_time_str:8s} | {cum_dist_str:8s} | {hr_str:11s} | {power_str:13s} | {cadence_str:15s} | {step_str:11s} | {gct_str:7s} | {elev_str}"
            print(row)
        
        # Top 3 mejores laps por pace (más rápidos) con métricas acumuladas
        if 'pace_calculated' in self.df_laps.columns:
            valid_laps = self.df_laps.dropna(subset=['pace_calculated'])
            if len(valid_laps) > 0:
                best_laps = valid_laps.nsmallest(3, 'pace_calculated')
                
                print(f"\n🥇 TOP 3 LAPS MÁS RÁPIDOS:")
                for idx, (lap_idx, lap) in enumerate(best_laps.iterrows(), 1):
                    pace_min = int(lap['pace_calculated'])
                    pace_sec = int((lap['pace_calculated'] % 1) * 60)
                    pace_str = f"{pace_min}:{pace_sec:02d}"
                    hr = int(lap['avg_heart_rate']) if pd.notna(lap['avg_heart_rate']) else 'N/A'
                    power = int(lap['avg_power']) if pd.notna(lap['avg_power']) else 'N/A'
                    
                    # Duración del lap individual
                    lap_duration = lap.get('total_timer_time', 0)
                    lap_duration_str = self._format_time(lap_duration)
                    
                    # Distancia del lap individual
                    lap_distance = lap.get('total_distance', 0) / 1000
                    lap_distance_str = f"{lap_distance:.1f}km"
                    
                    # Delta de altitud del lap individual
                    ascent = int(lap.get('total_ascent', 0))
                    descent = int(lap.get('total_descent', 0))
                    altitude_delta = ascent - descent
                    delta_str = f"{altitude_delta:+d}m" if altitude_delta != 0 else "±0m"
                    
                    print(f"   {idx}. Lap {lap_idx + 1}: {pace_str}/km | {hr} bpm | {power} W | {lap_duration_str} | {lap_distance_str} | {delta_str}")
    
    def _print_pauses_analysis(self):
        """4. Análisis de Pausas"""
        print(f"\n⏸️  ANÁLISIS DE PAUSAS")
        print("-" * 40)
        
        if not self.pauses:
            print("✅ Sesión continua sin pausas")
            return
        
        print(f"📊 Total de pausas: {len(self.pauses)}")
        
        for i, pause in enumerate(self.pauses, 1):
            distance = pause['distance_km']
            duration = pause['duration_formatted']
            
            # Clasificar tipo de pausa por duración
            seconds = pause['duration_seconds']
            if seconds < 30:
                pause_type = "Micro-pausa"
            elif seconds < 300:  # 5 min
                pause_type = "Pausa corta"  
            else:
                pause_type = "Pausa larga"
            
            print(f"   {i}. Km {distance:.1f} → {duration} ({pause_type})")
        
        # Tiempo total en pausa
        total_pause_time = sum(p['duration_seconds'] for p in self.pauses)
        if total_pause_time > 0:
            pause_formatted = str(timedelta(seconds=int(total_pause_time)))
            print(f"\n⏱️  Tiempo total en pausa: {pause_formatted}")
    
    def _print_equipment_summary(self):
        """5. Equipamiento"""
        print(f"\n📱 EQUIPAMIENTO")
        print("-" * 40)
        
        if not self.key_devices:
            print("Sin información de sensores externos")
            return
        
        device_types = {
            'heart_rate': '💓 Monitor cardíaco',
            'stride_speed_distance': '👟 Sensor de carrera'
        }
        
        for device in self.key_devices:
            device_type = device['type']
            device_name = device_types.get(device_type, f"📡 Sensor {device_type}")
            
            # Filtrar "None" o valores extraños
            if device_type is None or str(device_type).lower() == 'none':
                continue
                
            status_parts = [device_name]
            
            if device['battery_status'] == 'ok':
                if device['battery_voltage']:
                    voltage = device['battery_voltage']
                    status_parts.append(f"🔋 {voltage:.2f}V")
                else:
                    status_parts.append("🔋 OK")
            
            print(f"   {' | '.join(status_parts)}")
        
        if not any(device['type'] is not None and str(device['type']).lower() != 'none' 
                  for device in self.key_devices):
            print("   📱 Solo reloj GPS detectado")
    
    def _print_footer(self):
        """Footer del reporte"""
        print(f"\n" + "="*80)
        print("📈 Reporte generado por Dream FIT Analyzer")
        print("="*80)
    
    def export_data(self, output_dir="output"):
        """Exporta datos procesados con enfoque en análisis por lap"""
        Path(output_dir).mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        exported_files = []
        
        # 1. Análisis completo por lap con métricas acumuladas
        if self.df_laps is not None:
            lap_analysis_cols = [
                'total_timer_time', 'total_distance', 'pace_calculated',
                'cumulative_time', 'cumulative_distance',
                'avg_heart_rate', 'max_heart_rate', 
                'avg_power', 'max_power',
                'avg_running_cadence_corrected', 'max_running_cadence_corrected',
                'avg_step_length_m', 'gct_imbalance_ms',
                'avg_vertical_oscillation', 'avg_stance_time', 'avg_stance_time_balance',
                'total_ascent', 'total_descent',
                'avg_temperature', 'total_calories'
            ]
            
            available_cols = [col for col in lap_analysis_cols if col in self.df_laps.columns]
            lap_analysis = self.df_laps[available_cols].copy()
            lap_analysis.index += 1  # Laps empiezan en 1
            
            filename = f"detailed_lap_analysis_{timestamp}.csv"
            lap_analysis.to_csv(f"{output_dir}/{filename}")
            exported_files.append(f"{filename} - {len(lap_analysis)} laps con análisis completo")
        
        # 2. Records limpios con métricas corregidas
        if self.df_records is not None:
            record_cols = [
                'timestamp', 'elapsed_seconds', 'distance',
                'heart_rate', 'power', 
                'cadence_corrected', 'enhanced_speed', 'pace_min_km',
                'step_length_m', 'vertical_oscillation', 'vertical_ratio',
                'stance_time', 'stance_time_balance',
                'gct_left_ms', 'gct_right_ms', 'gct_imbalance_ms'
            ]
            
            available_cols = [col for col in record_cols if col in self.df_records.columns]
            clean_records = self.df_records[available_cols]
            
            filename = f"records_with_corrections_{timestamp}.csv"
            clean_records.to_csv(f"{output_dir}/{filename}", index=False)
            exported_files.append(f"{filename} - {len(clean_records)} registros corregidos")
        
        # 3. Resumen de pausas
        if self.pauses:
            pauses_df = pd.DataFrame(self.pauses)
            filename = f"pauses_summary_{timestamp}.csv"
            pauses_df.to_csv(f"{output_dir}/{filename}", index=False)
            exported_files.append(f"{filename} - {len(self.pauses)} pausas")
        
        # 4. Tabla de laps en formato visual mejorado
        if self.df_laps is not None:
            filename = f"lap_analysis_table_{timestamp}.txt"
            self._export_visual_lap_table(f"{output_dir}/{filename}")
            exported_files.append(f"{filename} - Tabla visual de laps")
        
        if exported_files:
            print(f"\n💾 ARCHIVOS EXPORTADOS:")
            for file_info in exported_files:
                print(f"   ✅ {file_info}")
            print(f"\n📁 Ubicación: {output_dir}/")
    
    def _export_visual_lap_table(self, filename):
        """Exporta tabla de laps en formato visual mejorado con métricas acumuladas"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("🏁 ANÁLISIS DETALLADO POR LAP\n")
            f.write("=" * 100 + "\n")
            f.write(f"Actividad: {os.path.basename(self.fit_file_path)}\n")
            
            if self.session_start_local:
                if self.timezone_offset is not None and self.timezone_offset != 0:
                    date_str = self.session_start_local.strftime('%d/%m/%Y %H:%M') + f" (UTC{self.timezone_offset:+d})"
                else:
                    date_str = self.session_start_local.strftime('%d/%m/%Y %H:%M (hora local)')
                f.write(f"Fecha: {date_str}\n")
            
            if self.location_info:
                f.write(f"Ubicación: {self.location_info}\n")
                
            f.write("\n")
            
            # Header de la tabla (expandido)
            header = "Lap | Pace    | Dist  | Cum Time | Cum Dist | HR Avg/Max | Power Avg/Max | Cadence Avg/Max | Step Length | GCT Imb | Elev +/-"
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")
            
            for i, (_, lap) in enumerate(self.df_laps.iterrows(), 1):
                # Pace
                if pd.notna(lap.get('pace_calculated')):
                    pace_min = int(lap['pace_calculated'])
                    pace_sec = int((lap['pace_calculated'] % 1) * 60)
                    pace_str = f"{pace_min}:{pace_sec:02d}"
                else:
                    pace_str = "N/A"
                
                # Distancia
                distance = lap.get('total_distance', 0) / 1000
                dist_str = f"{distance:.1f}km"
                
                # Tiempo acumulado
                cum_time = lap.get('cumulative_time', 0)
                cum_time_str = self._format_time(cum_time)
                
                # Distancia acumulada
                cum_dist = lap.get('cumulative_distance', 0) / 1000
                cum_dist_str = f"{cum_dist:.1f}km"
                
                # Heart Rate
                hr_avg = int(lap['avg_heart_rate']) if pd.notna(lap.get('avg_heart_rate')) else 'N/A'
                hr_max = int(lap['max_heart_rate']) if pd.notna(lap.get('max_heart_rate')) else 'N/A'
                hr_str = f"{hr_avg}/{hr_max}"
                
                # Power
                power_avg = int(lap['avg_power']) if pd.notna(lap.get('avg_power')) else 'N/A'
                power_max = int(lap['max_power']) if pd.notna(lap.get('max_power')) else 'N/A'
                power_str = f"{power_avg}/{power_max}"
                
                # Cadencia corregida
                cadence_avg = int(lap['avg_running_cadence_corrected']) if pd.notna(lap.get('avg_running_cadence_corrected')) else 'N/A'
                cadence_max = int(lap['max_running_cadence_corrected']) if pd.notna(lap.get('max_running_cadence_corrected')) else 'N/A'
                cadence_str = f"{cadence_avg}/{cadence_max}"
                
                # Step Length en metros
                step_length = lap.get('avg_step_length_m', 0)
                step_str = f"{step_length:.2f}m" if step_length > 0 else 'N/A'
                
                # GCT Imbalance
                gct_imb = lap.get('gct_imbalance_ms')
                gct_str = f"{gct_imb:.1f}ms" if pd.notna(gct_imb) else 'N/A'
                
                # Elevación
                ascent = int(lap.get('total_ascent', 0))
                descent = int(lap.get('total_descent', 0))
                elev_str = f"+{ascent}/-{descent}"
                
                # Formatear fila
                row = f"{i:2d}  | {pace_str:7s} | {dist_str:5s} | {cum_time_str:8s} | {cum_dist_str:8s} | {hr_str:11s} | {power_str:13s} | {cadence_str:15s} | {step_str:11s} | {gct_str:7s} | {elev_str}"
                f.write(row + "\n")
            
            # Top 3 mejores laps con métricas acumuladas
            if 'pace_calculated' in self.df_laps.columns:
                valid_laps = self.df_laps.dropna(subset=['pace_calculated'])
                if len(valid_laps) > 0:
                    best_laps = valid_laps.nsmallest(3, 'pace_calculated')
                    
                    f.write(f"\n🥇 TOP 3 LAPS MÁS RÁPIDOS:\n")
                    for idx, (lap_idx, lap) in enumerate(best_laps.iterrows(), 1):
                        pace_min = int(lap['pace_calculated'])
                        pace_sec = int((lap['pace_calculated'] % 1) * 60)
                        pace_str = f"{pace_min}:{pace_sec:02d}"
                        hr = int(lap['avg_heart_rate']) if pd.notna(lap['avg_heart_rate']) else 'N/A'
                        power = int(lap['avg_power']) if pd.notna(lap['avg_power']) else 'N/A'
                        
                        # Duración del lap individual
                        lap_duration = lap.get('total_timer_time', 0)
                        lap_duration_str = self._format_time(lap_duration)
                        
                        # Distancia del lap individual
                        lap_distance = lap.get('total_distance', 0) / 1000
                        lap_distance_str = f"{lap_distance:.1f}km"
                        
                        # Delta de altitud del lap individual
                        ascent = int(lap.get('total_ascent', 0))
                        descent = int(lap.get('total_descent', 0))
                        altitude_delta = ascent - descent
                        delta_str = f"{altitude_delta:+d}m" if altitude_delta != 0 else "±0m"
                        
                        f.write(f"   {idx}. Lap {lap_idx + 1}: {pace_str}/km | {hr} bpm | {power} W | {lap_duration_str} | {lap_distance_str} | {delta_str}\n")
            
            f.write(f"\n" + "=" * 100 + "\n")
            f.write("📋 Este archivo se puede copiar/pegar en Excel o Google Sheets\n")
            f.write("📊 Los datos también están disponibles en CSV para análisis avanzado\n")
            f.write("⏱️  Cum Time = Tiempo acumulado hasta ese lap\n")
            f.write("📏 Cum Dist = Distancia acumulada hasta ese lap\n")

def main():
    if len(sys.argv) != 2:
        print("Uso: python dream_report.py archivo.fit")
        sys.exit(1)
    
    fit_file = sys.argv[1]
    
    if not os.path.exists(fit_file):
        print(f"❌ Archivo no encontrado: {fit_file}")
        sys.exit(1)
    
    # Crear y ejecutar análisis
    analyzer = FitDreamAnalyzer(fit_file)
    
    if not analyzer.load_fit_file():
        sys.exit(1)
    
    analyzer.process_data()
    analyzer.generate_report()
    analyzer.export_data()
    
    if not REVERSE_GEOCODER_AVAILABLE:
        print("\n💡 TIP: Instala 'reverse_geocoder' para detección de ubicación:")
        print("   pip install reverse_geocoder")

if __name__ == "__main__":
    main()
