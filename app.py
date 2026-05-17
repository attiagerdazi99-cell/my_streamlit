# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time

st.set_page_config(
    page_title="CARE-X | Clinical Escalation Reasoning",
    page_icon="⚕️",
    layout="wide"
)

# Custom CSS for clinical feel
st.markdown("""
<style>
    .clinician-note {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2c3e50;
        margin: 1rem 0;
    }
    .uncertainty-low {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .uncertainty-moderate {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .uncertainty-high {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .uncertainty-critical {
        background-color: #d1ecf1;
        border-left: 4px solid #17a2b8;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .reasoning-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .escalation-pathway {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .warning-sign {
        background-color: #fff3cd;
        border-left: 4px solid #ff9800;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CORE CLINICAL REASONING ENGINES
# ============================================================================

class TrendBasedDeteriorationEngine:
    """Detects evolving instability patterns, not isolated abnormalities"""
    
    @staticmethod
    def analyze(physiological_history):
        """Prioritizes trends over isolated values"""
        if len(physiological_history) < 2:
            return {}
        
        latest = physiological_history[-1]
        initial = physiological_history[0]
        previous = physiological_history[-2] if len(physiological_history) > 1 else latest
        
        trends = {
            'sbp_trend': latest['sbp'] - previous['sbp'],
            'sbp_cumulative': latest['sbp'] - initial['sbp'],
            'lactate_trend': latest['lactate'] - previous['lactate'],
            'lactate_cumulative': latest['lactate'] - initial['lactate'],
            'hr_trend': latest['hr'] - previous['hr'],
            'hr_cumulative': latest['hr'] - initial['hr'],
            'oxygen_trend': latest['oxygen'] - previous['oxygen'],
            'oxygen_cumulative': latest['oxygen'] - initial['oxygen']
        }
        
        # Pattern recognition
        patterns = []
        
        # Evolving hypotension
        if trends['sbp_cumulative'] < -15:
            patterns.append("progressive hypotension")
        elif trends['sbp_cumulative'] < -10:
            patterns.append("evolving hypotension")
        
        # Rising lactate pattern
        if trends['lactate_cumulative'] > 1.5:
            patterns.append("significantly rising lactate")
        elif trends['lactate_cumulative'] > 0.8:
            patterns.append("persistently rising lactate")
        
        # Tachycardia despite compensation
        if latest['hr'] > 105 and trends['hr_trend'] > 5:
            patterns.append("worsening tachycardia despite compensation")
        
        # Increasing oxygen requirement
        if trends['oxygen_cumulative'] > 2:
            patterns.append("progressive oxygen requirement")
        elif trends['oxygen_trend'] > 0 and latest['oxygen'] > 2:
            patterns.append("increasing oxygen requirement")
        
        worsening_compensation = any([
            latest['sbp'] < 95 and latest['hr'] > 105,
            trends['lactate_trend'] > 0.3 and latest['lactate'] > 2,
            trends['oxygen_trend'] > 0.5 and latest['hr'] > 105
        ])
        
        return {
            'trends': trends,
            'patterns': patterns,
            'evolving_instability': len(patterns) >= 1,
            'worsening_compensation': worsening_compensation,
            'progressive_stress': trends['lactate_cumulative'] > 1 or trends['sbp_cumulative'] < -15
        }


class EscalationUncertaintyModel:
    """Models clinical uncertainty zones where escalation hesitation occurs"""
    
    @staticmethod
    def assess(physiological_history, trends_analysis):
        if len(physiological_history) == 0:
            return {'level': 'LOW', 'confidence': 0, 'uncertainty_zones': []}
        
        latest = physiological_history[-1]
        
        # Uncertainty zone detection
        uncertainty_zones = []
        score = 0
        
        # Borderline hypotension zone
        if 85 < latest['sbp'] < 95:
            uncertainty_zones.append("borderline hypotension")
            score += 1
        elif latest['sbp'] <= 85:
            score += 3
        
        # Mild but rising lactate zone
        if 1.5 < latest['lactate'] < 3.0:
            if trends_analysis.get('trends', {}).get('lactate_trend', 0) > 0:
                uncertainty_zones.append("mild but rising lactate")
                score += 2
        elif latest['lactate'] >= 3.0:
            score += 3
        
        # Mixed picture zone
        if latest['sbp'] < 100 and latest['lactate'] < 2.5 and latest['hr'] > 105:
            uncertainty_zones.append("mixed sepsis/cardiac picture")
            score += 2
        
        # Transient compensation zone
        if trends_analysis.get('worsening_compensation', False):
            uncertainty_zones.append("transient compensation with underlying deterioration")
            score += 2
        
        # Incomplete shock criteria
        shock_criteria_met = sum([
            latest['sbp'] < 90,
            latest['lactate'] > 2,
            latest['hr'] > 110
        ])
        if 1 <= shock_criteria_met <= 2:
            uncertainty_zones.append("incomplete overt shock criteria")
            score += 1
        
        # Non-specific deterioration
        if len(trends_analysis.get('patterns', [])) >= 2 and shock_criteria_met == 0:
            uncertainty_zones.append("non-specific deterioration pattern")
            score += 1
        
        # Escalation confidence (inverse of uncertainty)
        if score <= 2:
            level = 'LOW'
            confidence = 85
            message = "Current abnormalities remain clinically reassuring. No immediate escalation indicated."
        elif score <= 5:
            level = 'MODERATE'
            confidence = 60
            message = "Current findings may represent evolving instability, but overt shock criteria are incomplete. Escalation consideration is reasonable."
        elif score <= 8:
            level = 'HIGH'
            confidence = 35
            message = "Progressive abnormalities increase concern for delayed decompensation despite transient compensation. Escalation should be strongly considered."
        else:
            level = 'CRITICAL'
            confidence = 15
            message = "Persistent deterioration patterns suggest impending decompensation. Immediate escalation recommended."
        
        return {
            'score': score,
            'level': level,
            'confidence': confidence,
            'message': message,
            'uncertainty_zones': uncertainty_zones
        }


class ClinicalInterpretationEngine:
    """Generates nuanced clinical interpretations that feel like real clinician cognition"""
    
    @staticmethod
    def generate(physiological_history, trends_analysis, uncertainty):
        if len(physiological_history) == 0:
            return "Insufficient data for clinical interpretation."
        
        latest = physiological_history[-1]
        interpretations = []
        
        # Hemodynamic assessment
        if latest['sbp'] < 90 and trends_analysis.get('trends', {}).get('sbp_cumulative', 0) < -10:
            interpretations.append("This patient demonstrates evolving hemodynamic instability with progressive hypotension despite compensatory mechanisms.")
        elif latest['sbp'] < 95:
            interpretations.append("Borderline hypotension with downward trajectory suggests worsening cardiovascular compensation.")
        
        # Perfusion assessment
        if latest['lactate'] > 3:
            interpretations.append("Elevated lactate with rising trend indicates worsening tissue hypoperfusion.")
        elif latest['lactate'] > 2:
            interpretations.append("Persistent lactate elevation, although not yet critical, raises concern for evolving hypoperfusion.")
        
        # Integration
        if uncertainty['level'] == 'MODERATE':
            interpretations.append("Although overt shock is not yet fully established, the current pattern raises concern for delayed cardiovascular decompensation.")
        elif uncertainty['level'] == 'HIGH':
            interpretations.append("Persistent abnormalities despite transient physiological compensation may justify earlier escalation before criteria become overt.")
        
        if trends_analysis.get('worsening_compensation', False):
            interpretations.append("Worsening tachycardia despite fluid resuscitation suggests compensation is failing.")
        
        return " ".join(interpretations) if interpretations else "Vital signs remain within range, but ongoing monitoring warranted."


class EscalationRationaleEngine:
    """Explains WHY escalation consideration may already be appropriate despite incomplete criteria"""
    
    @staticmethod
    def generate(physiological_history, trends_analysis, uncertainty):
        if len(physiological_history) < 2:
            return ["Insufficient trend data for escalation rationale."]
        
        latest = physiological_history[-1]
        previous = physiological_history[-2]
        rationale = []
        
        # Key reasoning points
        if 85 < latest['sbp'] < 95:
            rationale.append("Borderline hypotension combined with downward trend may represent early cardiovascular deterioration before overt shock develops.")
        
        if latest['lactate'] > 2 and trends_analysis.get('trends', {}).get('lactate_trend', 0) > 0:
            rationale.append("Rising lactate despite borderline blood pressure suggests evolving tissue hypoperfusion that may not yet be clinically apparent.")
        
        if latest['hr'] > 105 and previous['hr'] > 100:
            rationale.append("Persistent tachycardia despite initial management indicates ongoing physiological stress and compensation failure.")
        
        if trends_analysis.get('trends', {}).get('oxygen_trend', 0) > 0:
            rationale.append("Increasing oxygen requirement signals progressive physiological deterioration beyond initial insult.")
        
        if uncertainty['level'] in ['HIGH', 'MODERATE']:
            rationale.append("Delayed escalation until overt shock criteria develop may increase risk of sudden decompensation and worse outcomes.")
        
        if uncertainty.get('uncertainty_zones'):
            rationale.append(f"Clinical ambiguity exists in this case ({', '.join(uncertainty['uncertainty_zones'][:2])}), making early escalation consideration prudent.")
        
        return rationale


class DynamicEscalationEngine:
    """Progressive escalation recommendations that evolve with clinical trajectory"""
    
    @staticmethod
    def determine(physiological_history, uncertainty, trends_analysis):
        level = uncertainty['level']
        
        if level == 'LOW':
            return {
                'pathway': 'Continued Monitoring',
                'urgency': 'Routine',
                'timeline': 'Reassess in 2-4 hours',
                'actions': [
                    'Continue current monitoring protocol',
                    'Repeat lactate in 4 hours',
                    'Clinical review in morning'
                ],
                'escalation_threshold': 'Worsening trend or new symptoms'
            }
        
        elif level == 'MODERATE':
            return {
                'pathway': 'Senior Clinical Review',
                'urgency': 'Urgent (within 1-2 hours)',
                'timeline': 'Reassess within 1 hour',
                'actions': [
                    'Urgent physician review at bedside',
                    'Reduce vital sign monitoring to q30min',
                    'Consider senior registrar discussion',
                    'Repeat lactate in 1-2 hours',
                    'Consider early ICU liaison review'
                ],
                'escalation_threshold': 'Persistent worsening or clinical deterioration'
            }
        
        elif level == 'HIGH':
            return {
                'pathway': 'ICU Escalation Consideration',
                'urgency': 'Immediate senior review',
                'timeline': 'Reassess within 30 minutes',
                'actions': [
                    'Immediate ICU consultation',
                    'Prepare for potential vasopressor support',
                    'Establish IV access if not present',
                    'Consider arterial line for monitoring',
                    'Notify intensive care registrar'
                ],
                'escalation_threshold': 'Any further deterioration triggers ICU transfer'
            }
        
        else:  # CRITICAL
            return {
                'pathway': 'Emergency Critical Care Activation',
                'urgency': 'Immediate',
                'timeline': 'Continuous reassessment',
                'actions': [
                    'ACTIVATE emergency critical care response',
                    'Prepare for immediate ICU transfer',
                    'Initiate vasopressor support protocol',
                    'Secure definitive airway if indicated',
                    'Activate MET/code blue team if appropriate'
                ],
                'escalation_threshold': 'Patient requires immediate critical care'
            }


class DynamicWhatIfSimulator:
    """Real-time simulation showing how escalation reasoning evolves with different trajectories"""
    
    @staticmethod
    def simulate(current_state, scenario_type):
        latest = current_state[-1] if current_state else {'sbp': 95, 'lactate': 2.5, 'hr': 110, 'oxygen': 2}
        
        scenarios = {
            'improving_sbp': {
                'sbp': min(120, latest['sbp'] + 10),
                'lactate': latest['lactate'],
                'hr': max(80, latest['hr'] - 5),
                'oxygen': latest['oxygen']
            },
            'worsening_lactate': {
                'sbp': latest['sbp'],
                'lactate': latest['lactate'] + 1.0,
                'hr': min(140, latest['hr'] + 8),
                'oxygen': latest['oxygen']
            },
            'worsening_oxygen': {
                'sbp': latest['sbp'],
                'lactate': latest['lactate'] + 0.3,
                'hr': min(140, latest['hr'] + 5),
                'oxygen': latest['oxygen'] + 2
            },
            'persistent_tachycardia': {
                'sbp': latest['sbp'],
                'lactate': latest['lactate'] + 0.2,
                'hr': latest['hr'],
                'oxygen': latest['oxygen'] + 0.5
            }
        }
        
        scenario = scenarios.get(scenario_type, scenarios['worsening_lactate'])
        
        # Create temporary history for simulation
        sim_history = current_state + [{
            'time': datetime.now(),
            'sbp': scenario['sbp'],
            'hr': scenario['hr'],
            'lactate': scenario['lactate'],
            'oxygen': scenario['oxygen'],
            'resp_rate': latest.get('resp_rate', 18) + 2
        }]
        
        # Analyze simulated trajectory
        trend_engine = TrendBasedDeteriorationEngine()
        uncertainty_engine = EscalationUncertaintyModel()
        
        sim_trends = trend_engine.analyze(sim_history)
        sim_uncertainty = uncertainty_engine.assess(sim_history, sim_trends)
        
        return {
            'scenario': scenario_type.replace('_', ' ').title(),
            'values': scenario,
            'uncertainty_level': sim_uncertainty['level'],
            'confidence_change': 65 - sim_uncertainty['confidence'],
            'interpretation': f"If {scenario_type.replace('_', ' ')} occurs, escalation confidence would change to {sim_uncertainty['level']} level.",
            'recommendation': "Escalation would become more urgent" if sim_uncertainty['level'] in ['HIGH', 'CRITICAL'] else "Current monitoring may remain appropriate"
        }


class EscalationConfidenceTracker:
    """Tracks how escalation confidence evolves over time"""
    
    @staticmethod
    def track_evolution(physiological_history):
        if len(physiological_history) < 3:
            return []
        
        confidence_history = []
        for i in range(2, len(physiological_history) + 1):
            subset = physiological_history[:i]
            trend_engine = TrendBasedDeteriorationEngine()
            uncertainty_engine = EscalationUncertaintyModel()
            
            trends = trend_engine.analyze(subset)
            uncertainty = uncertainty_engine.assess(subset, trends)
            
            confidence_history.append({
                'time': subset[-1]['time'],
                'confidence': uncertainty['confidence'],
                'level': uncertainty['level']
            })
        
        return confidence_history


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def initialize_session_state():
    if 'physiological_history' not in st.session_state:
        # Initial patient state - borderline instability
        start_time = datetime.now() - timedelta(hours=4)
        st.session_state.physiological_history = [
            {
                'time': start_time + timedelta(hours=0),
                'sbp': 108, 'hr': 102, 'lactate': 2.1, 'oxygen': 2, 'resp_rate': 18
            },
            {
                'time': start_time + timedelta(hours=1),
                'sbp': 101, 'hr': 108, 'lactate': 2.5, 'oxygen': 2, 'resp_rate': 20
            },
            {
                'time': start_time + timedelta(hours=2),
                'sbp': 95, 'hr': 115, 'lactate': 2.9, 'oxygen': 3, 'resp_rate': 22
            },
            {
                'time': start_time + timedelta(hours=3),
                'sbp': 92, 'hr': 121, 'lactate': 3.2, 'oxygen': 4, 'resp_rate': 24
            },
            {
                'time': start_time + timedelta(hours=4),
                'sbp': 88, 'hr': 128, 'lactate': 3.6, 'oxygen': 5, 'resp_rate': 26
            }
        ]
    
    if 'simulation_active' not in st.session_state:
        st.session_state.simulation_active = False


def add_new_observation(sbp, hr, lactate, oxygen, resp_rate):
    new_obs = {
        'time': datetime.now(),
        'sbp': sbp,
        'hr': hr,
        'lactate': lactate,
        'oxygen': oxygen,
        'resp_rate': resp_rate
    }
    st.session_state.physiological_history.append(new_obs)
    
    # Keep last 20 observations
    if len(st.session_state.physiological_history) > 20:
        st.session_state.physiological_history = st.session_state.physiological_history[-20:]


def main():
    initialize_session_state()
    
    # Header
    st.title("⚕️ CARE-X")
    st.caption("Clinical Escalation Reasoning — Dynamic Instability Assessment")
    st.markdown("---")
    
    # Current patient context
    current = st.session_state.physiological_history[-1]
    previous = st.session_state.physiological_history[-2] if len(st.session_state.physiological_history) > 1 else current
    
    # Sidebar - Clinical Input
    with st.sidebar:
        st.header("📝 New Clinical Observation")
        
        col1, col2 = st.columns(2)
        with col1:
            new_sbp = st.number_input("SBP (mmHg)", min_value=60, max_value=200, value=int(current['sbp']))
            new_hr = st.number_input("Heart Rate", min_value=40, max_value=180, value=int(current['hr']))
        with col2:
            new_lactate = st.number_input("Lactate (mmol/L)", min_value=0.5, max_value=15.0, value=float(current['lactate']), step=0.1)
            new_oxygen = st.number_input("O₂ Requirement (L/min)", min_value=0, max_value=15, value=int(current['oxygen']))
        
        new_resp_rate = st.number_input("Respiratory Rate", min_value=8, max_value=50, value=int(current['resp_rate']))
        
        if st.button("➕ Add Observation & Update Reasoning", type="primary"):
            add_new_observation(new_sbp, new_hr, new_lactate, new_oxygen, new_resp_rate)
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 🎯 What-If Simulation")
        
        sim_options = ['worsening_lactate', 'worsening_oxygen', 'improving_sbp', 'persistent_tachycardia']
        selected_sim = st.selectbox("Select scenario:", sim_options)
        
        if st.button("🔮 Run Simulation"):
            st.session_state.simulation_active = selected_sim
    
    # Main display area
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate current clinical reasoning
    trend_engine = TrendBasedDeteriorationEngine()
    uncertainty_engine = EscalationUncertaintyModel()
    interpretation_engine = ClinicalInterpretationEngine()
    rationale_engine = EscalationRationaleEngine()
    escalation_engine = DynamicEscalationEngine()
    confidence_tracker = EscalationConfidenceTracker()
    
    trends = trend_engine.analyze(st.session_state.physiological_history)
    uncertainty = uncertainty_engine.assess(st.session_state.physiological_history, trends)
    interpretation = interpretation_engine.generate(st.session_state.physiological_history, trends, uncertainty)
    rationale = rationale_engine.generate(st.session_state.physiological_history, trends, uncertainty)
    escalation = escalation_engine.determine(st.session_state.physiological_history, uncertainty, trends)
    confidence_evolution = confidence_tracker.track_evolution(st.session_state.physiological_history)
    
    # Metrics
    with col1:
        delta_sbp = current['sbp'] - previous['sbp']
        st.metric("Current SBP", f"{current['sbp']} mmHg", f"{delta_sbp:+.0f}")
    
    with col2:
        delta_lactate = current['lactate'] - previous['lactate']
        st.metric("Current Lactate", f"{current['lactate']} mmol/L", f"{delta_lactate:+.1f}")
    
    with col3:
        delta_hr = current['hr'] - previous['hr']
        st.metric("Heart Rate", f"{current['hr']} bpm", f"{delta_hr:+.0f}")
    
    with col4:
        st.metric("Escalation Confidence", f"{uncertainty['confidence']}%", 
                  f"{uncertainty['confidence'] - 85 if confidence_evolution else 0:+.0f}%")
    
    # Escalation Uncertainty Zone (Primary display)
    st.markdown("---")
    st.subheader("🎯 Escalation Uncertainty Assessment")
    
    uncertainty_class = {
        'LOW': 'uncertainty-low',
        'MODERATE': 'uncertainty-moderate',
        'HIGH': 'uncertainty-high',
        'CRITICAL': 'uncertainty-critical'
    }.get(uncertainty['level'], 'uncertainty-moderate')
    
    st.markdown(f"""
    <div class="{uncertainty_class}">
        <strong>UNCERTAINTY LEVEL: {uncertainty['level']}</strong><br>
        {uncertainty['message']}
    </div>
    """, unsafe_allow_html=True)
    
    if uncertainty['uncertainty_zones']:
        st.markdown("**🔍 Identified Uncertainty Zones:**")
        for zone in uncertainty['uncertainty_zones']:
            st.markdown(f"- {zone}")
    
    # Clinical Interpretation
    st.markdown("---")
    st.subheader("🧠 Clinical Interpretation")
    st.markdown(f'<div class="clinician-note">{interpretation}</div>', unsafe_allow_html=True)
    
    # Escalation Rationale
    st.subheader("📋 Why Escalation Deserves Consideration")
    for r in rationale:
        st.markdown(f'<div class="warning-sign">• {r}</div>', unsafe_allow_html=True)
    
    # Escalation Pathway
    st.subheader("🚨 Escalation Pathway")
    st.markdown(f"""
    <div class="escalation-pathway">
        <strong>PATHWAY:</strong> {escalation['pathway']}<br>
        <strong>URGENCY:</strong> {escalation['urgency']}<br>
        <strong>TIMELINE:</strong> {escalation['timeline']}
    </div>
    """, unsafe_allow_html=True)
    
    for action in escalation['actions']:
        st.markdown(f"- {action}")
    
    # Trend Visualization
    st.markdown("---")
    st.subheader("📈 Evolving Deterioration Patterns")
    
    df_history = pd.DataFrame(st.session_state.physiological_history)
    df_history['time_str'] = df_history['time'].dt.strftime('%H:%M')
    
    fig = make_subplots(rows=2, cols=2, subplot_titles=('SBP Trend', 'Lactate Trend', 'Heart Rate Trend', 'O₂ Requirement'))
    
    fig.add_trace(go.Scatter(x=df_history['time_str'], y=df_history['sbp'], mode='lines+markers', name='SBP', line=dict(color='red', width=2)), row=1, col=1)
    fig.add_hline(y=90, line_dash="dash", line_color="red", row=1, col=1, annotation_text="Shock threshold")
    
    fig.add_trace(go.Scatter(x=df_history['time_str'], y=df_history['lactate'], mode='lines+markers', name='Lactate', line=dict(color='orange', width=2)), row=1, col=2)
    fig.add_hline(y=2.0, line_dash="dash", line_color="orange", row=1, col=2, annotation_text="Normal upper limit")
    
    fig.add_trace(go.Scatter(x=df_history['time_str'], y=df_history['hr'], mode='lines+markers', name='HR', line=dict(color='purple', width=2)), row=2, col=1)
    fig.add_hline(y=100, line_dash="dash", line_color="purple", row=2, col=1, annotation_text="Tachycardia")
    
    fig.add_trace(go.Scatter(x=df_history['time_str'], y=df_history['oxygen'], mode='lines+markers', name='O₂', line=dict(color='blue', width=2)), row=2, col=2)
    
    fig.update_layout(height=600, showlegend=False, title_text="Physiological Trajectory")
    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="mmHg", row=1, col=1)
    fig.update_yaxes(title_text="mmol/L", row=1, col=2)
    fig.update_yaxes(title_text="bpm", row=2, col=1)
    fig.update_yaxes(title_text="L/min", row=2, col=2)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Pattern recognition
    if trends['patterns']:
        st.markdown("**⚠️ Detected Deterioration Patterns:**")
        for pattern in trends['patterns']:
            st.markdown(f"- {pattern}")
    
    # Confidence evolution
    if confidence_evolution:
        st.markdown("---")
        st.subheader("📊 Escalation Confidence Evolution")
        
        conf_df = pd.DataFrame(confidence_evolution)
        conf_df['time_str'] = pd.to_datetime(conf_df['time']).dt.strftime('%H:%M')
        
        fig_conf = go.Figure()
        fig_conf.add_trace(go.Scatter(x=conf_df['time_str'], y=conf_df['confidence'], mode='lines+markers', 
                                       name='Confidence', line=dict(color='green', width=3)))
        fig_conf.add_hline(y=75, line_dash="dash", line_color="yellow", annotation_text="Moderate confidence threshold")
        fig_conf.add_hline(y=50, line_dash="dash", line_color="orange", annotation_text="High uncertainty threshold")
        fig_conf.update_layout(title="Escalation Confidence Over Time (Lower = Higher Concern)", 
                               xaxis_title="Time", yaxis_title="Confidence (%)", height=300)
        st.plotly_chart(fig_conf, use_container_width=True)
    
    # What-If Simulation Display
    if st.session_state.simulation_active:
        st.markdown("---")
        st.subheader("🔮 Dynamic What-If Simulation")
        
        sim_result = DynamicWhatIfSimulator.simulate(
            st.session_state.physiological_history, 
            st.session_state.simulation_active
        )
        
        st.markdown(f"""
        <div class="reasoning-box">
            <strong>Scenario:</strong> {sim_result['scenario']}<br>
            <strong>Projected Uncertainty Level:</strong> {sim_result['uncertainty_level']}<br>
            <strong>{sim_result['interpretation']}</strong><br>
            <strong>Clinical Implication:</strong> {sim_result['recommendation']}
        </div>
        """, unsafe_allow_html=True)
        
        if sim_result['uncertainty_level'] in ['HIGH', 'CRITICAL']:
            st.warning("⚠️ This trajectory would warrant immediate escalation consideration")
    
    # Clinical Workflow Context
    st.markdown("---")
    st.caption("⚕️ CARE-X is a clinical reasoning support tool. All escalation decisions require clinical judgment and senior review.")
    st.caption("Focus: Evolving instability • Escalation uncertainty • Delayed decompensation risk • Clinical ambiguity")


if __name__ == "__main__":
    main()
