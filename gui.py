import numpy as np
import pandas as pd
import streamlit as st

def simulate(cue, 
             predW,
             dt=2, 
             T=12*1000,  
             tau1=3, 
             t1r=11.25,
             sigma1=.5,
             sigma1r=1,
             tauE1=.015, 
             tauS1=.01,
             E1r=70,
             S1r=70,
             tau2=7.5, 
             sigma2=.1,
             sigma2r=1,
             tauE2=5, 
             tauS2=10, 
             E2r=10,
             S2r=5,
             tau3=100,
             sigma3=.025,
             tauE3=100,
             tauS3=50*150, 
             p=1.5,
             b=.1, b2=.1):
    
    tau1 *= 10
    tauE1 *= 10
    tauS1 *= 10
    tau2 *= 10
    tauE2 *= 10
    tauS2 *= 10
    tau3 *= 10
    tauE3 *= 10
    tauS3 *= 10
    
    # initialize 
    nt = T // dt
    tempWE1 = np.exp(-np.arange(0, nt)*dt/tauE1) * dt/tauE1  
    tempWS1 = np.exp(-np.arange(0, nt)*dt/tauS1) * dt/tauS1 
    tempWE1_r = np.exp(-np.arange(0, nt)*dt/(tauE1*E1r)) * dt/(tauE1*E1r) 
    tempWS1_r = np.exp(-np.arange(0, nt)*dt/(tauS1*E1r)) * dt/(tauS1*S1r)
    tempWE2 = np.exp(-np.arange(0, nt)*dt/(tauE2)) * dt/(tauE2)  
    tempWS2 = np.exp(-np.arange(0, nt)*dt/(tauS2)) * dt/(tauS2) 
    tempWE2_r = np.exp(-np.arange(0, nt)*dt/(tauE2*E2r)) * dt/(tauE2*E2r)
    tempWS2_r = np.exp(-np.arange(0, nt)*dt/(tauS2*S2r)) * dt/(tauS2*S2r)
    tempWE3 = np.exp(-np.arange(0, nt)*dt/tauE3) * dt/tauE3
    tempWS3 = np.exp(-np.arange(0, nt)*dt/tauS3) * dt/tauS3
    drive1, d1, s1, f1, r1 = np.zeros((5, nt)), np.zeros((5, nt)), np.zeros((5, nt)), np.zeros((5, nt)), np.zeros((5, nt))
    drive2, d2, s2, f2, r2 = np.zeros((5, nt)), np.zeros((5, nt)), np.zeros((5, nt)), np.zeros((5, nt)), np.zeros((5, nt))
    drive3, d3, s3, f3, r3 = np.zeros(nt), np.zeros(nt), np.zeros(nt), np.zeros(nt), np.zeros(nt)
    
    # simulate
    for i in range(1, nt):
        drive1[:, i] = cue[:, i]**p
        d1[:4, i] = np.sum(drive1[:4, :i] * tempWE1[i-1::-1], axis=1) 
        d1[4, i] = np.sum(drive1[4, :i] * tempWE1_r[i-1::-1]) 
        s1[:4, i] = np.sum(np.abs(d1[:4, :i]) * tempWS1[i-1::-1], axis=1) 
        s1[4, i] = np.sum(np.abs(d1[4, :i]) * tempWS1_r[i-1::-1]) 
        f1[:4, i] = d1[:4, i] / (np.abs(s1[:4, i]) + (sigma1)**p) 
        f1[4, i] = d1[4, i] / (np.abs(s1[4, i]) + (sigma1r)**p) 
        r1[:4, i] = r1[:4, i-1] + dt/(tau1)*(-r1[:4, i-1]+f1[:4, i]) 
        r1[4, i] = r1[4, i-1] + dt/(tau1*t1r)*(-r1[4, i-1]+f1[4, i]) 

        drive2[:4, i] = (r1[:4, i]/10)**p
        drive2[4, i] = (r1[4, i])**p
        d2[:4, i] = np.sum(drive2[:4, :i] * tempWE2[i-1::-1], axis=1) 
        d2[4, i] = np.sum(drive2[4, :i] * tempWE2_r[i-1::-1]) 
        s2[:4, i] = np.sum(np.abs(d2[:4, :i]) * tempWS2[i-1::-1], axis=1) 
        s2[4, i] = np.sum(np.abs(d2[4, :i]) * tempWS2_r[i-1::-1]) 
        f2[:4, i] = (d2[:4, i] + b) / (np.abs(s2[:4, i]).sum() + (sigma2)**p) 
        f2[4, i] = (d2[4, i] + b2) / (np.abs(s2[4, i]).sum() + (sigma2r)**p) 
        r2[:4, i] = r2[:4, i-1] + dt/(tau2)*(-r2[:4, i-1]+f2[:4, i]) 
        r2[4, i] = r2[4, i-1] + dt/(tau2*2)*(-r2[4, i-1]+f2[4, i]) 
        
        
        drive3[i] = max(0, predW @ r2[:, i])**p 
        d3[i] = np.sum(drive3[:i] * tempWE3[i-1::-1]) 
        s3[i] = np.sum(np.abs(d3[:i]) * tempWS3[i-1::-1])
        f3[i] = (d3[i]) / (s3[i] + sigma3**p) 
        r3[i] = r3[i-1] + dt/tau3*(-r3[i-1]+f3[i]) 
    

    return r1, r2, r3 

def wrap_simulate(args):
    tau1, t1r, sigma1, sigma1r, tauE1, tauS1, E1r, S1r, tau2, sigma2, sigma2r, tauE2, tauS2, E2r, S2r, tau3, sigma3, tauE3, tauS3, m, w1, w2, w3, w4, scale_rwd, b, b2 = args 
    cue = np.zeros((5, 644))
    cue[0, 244:294] = 1 
    dt = 4
    dt_psth = 20
    T = 644*dt_psth
    nt = T // dt
    p = 1.5
    predW = np.array([w1, w2, w3, w4, scale_rwd])

    soas = [0, 1500//dt, 3000//dt, 6000//dt]
    r1s = []
    r2s = []
    r3s = []

    for i, soa in enumerate(soas):
        cue = np.zeros((5, nt))
        cue_onset = 244*dt_psth//dt
        cue_offset = 294*dt_psth//dt
        rwd_onset = cue_onset + soa
        rwd_offset = rwd_onset + 1000//dt
        cue[i, cue_onset:cue_offset] = 1 #294
        cue[4, rwd_onset:rwd_offset] = 1
        r1, r2, r3 = simulate(cue, predW, tau1=tau1, sigma1=sigma1, sigma1r=sigma1r, tauE1=tauE1, tauS1=tauS1, E1r=E1r, S1r=S1r, tau2=tau2, sigma2=sigma2, sigma2r=sigma2r, tauE2=tauE2, tauS2=tauS2, E2r=E2r, S2r=S2r, tau3=tau3, sigma3=sigma3, tauE3=tauE3, tauS3=tauS3, dt=dt, T=T, p=p, b=b, b2=b2) 
        r3 = m*r3
        r1s.append(r1)
        r2s.append(r2)
        r3s.append(r3)

    return r1s, r2s, r3s

if "params_set" not in st.session_state:
    st.session_state["params_set"] = True
    tau1 = 3 
    t1r = 11.25
    sigma1 = .5
    sigma1r = .5
    tauE1 = .015 
    tauS1 = .01
    E1r = 70
    S1r = 70
    tau2 = 7.5 
    sigma2 = .1
    sigma2r = .1
    tauE2 = 5 
    tauS2 = 10 
    E2r = 10
    S2r = 5
    tau3 = 2
    sigma3 = 2.6
    tauE3 = 3
    tauS3 = 6.2
    w1 = 5.8
    w2 = 3.3
    w3 = .001
    w4 = 2.5
    scale_rwd = 20.0
    b = .04
    b2 = .00005
    m = 5.5
else:
    tau1 = st.session_state.tau1
    t1r = st.session_state.t1r
    sigma1 = st.session_state.sigma1
    sigma1r = st.session_state.sigma1r
    tauE1 = st.session_state.tauE1
    tauS1 = st.session_state.tauS1
    E1r = st.session_state.E1r
    S1r = st.session_state.S1r
    tau2 = st.session_state.tau2
    sigma2 = st.session_state.sigma2
    sigma2r = st.session_state.sigma2r
    tauE2 = st.session_state.tauE2
    tauS2 = st.session_state.tauS2
    E2r = st.session_state.E2r
    S2r = st.session_state.S2r
    tau3 = st.session_state.tau3
    sigma3 = st.session_state.sigma3
    tauE3 = st.session_state.tauE3
    tauS3 = st.session_state.tauS3
    w1 = st.session_state.w1
    w2 = st.session_state.w2
    w3 = st.session_state.w3
    w4 = st.session_state.w4
    scale_rwd = st.session_state.scale_rwd
    b = st.session_state.b
    b2 = st.session_state.b2
    m = st.session_state.m

r1s, r2s, r3s = wrap_simulate((tau1, t1r, sigma1, sigma1r, tauE1, tauS1, E1r, S1r, tau2, sigma2, sigma2r, tauE2, tauS2, E2r, S2r, tau3, sigma3, tauE3, tauS3, m, w1, w2, w3, w4, scale_rwd, b, b2))
df1 = pd.DataFrame({"S1-cue": r1s[0][0], "S1-rwd": r1s[2][-1]})
df2 = pd.DataFrame({"S2-cue": r2s[0][0], "S2-rwd": r2s[2][-1]})
df3 = pd.DataFrame({"R-cue1": r3s[0], "R-cue2": r3s[1], "R-cue3": r3s[2], "R-cue4": r3s[3]})


st.markdown("# Sensory 1 (S1)")

tau1 = st.slider(r"$\tau^\text{cue}_R$", min_value=1.0, max_value=100.0, value=3.0, step=1.0, key="tau1")
t1r = st.slider(r"$\frac{\tau^\text{rwd}_R}{\tau^\text{cue}_R}$ (i.e., $\tau_R$ for S1-reward neurons relative to S1-cue neurons)", min_value=1.0, max_value=100.0, value=11.25, step=1.0, key="t1r")
sigma1 = st.slider(r"$\sigma^\text{cue}$", min_value=.01, max_value=100.0, value=.5, step=.01, key="sigma1")
sigma1r = st.slider(r"$\sigma^\text{cue}$", min_value=.01, max_value=100.0, value=.5, step=.01, key="sigma1r")
tauE1 = st.slider(r"$\tau^\text{cue}_E$", min_value=.01, max_value=100.0, value=.015, step=.01, key="tauE1")
tauS1 = st.slider(r"$\tau^\text{cue}_S$", min_value=.01, max_value=100.0, value=.01, step=.01, key="tauS1")
E1r = st.slider(r"$\frac{\tau^\text{rwd}_E}{\tau^\text{cue}_E}$", min_value=1.0, max_value=1000.0, value=70.0, step=1.0, key="E1r")
S1r = st.slider(r"$\frac{\tau^\text{rwd}_S}{\tau^\text{cue}_S}$", min_value=1.0, max_value=1000.0, value=70.0, step=1.0, key="S1r")

st.line_chart(df1)


st.markdown("# Sensory 2 (S2)")

tau2 = st.slider(r"$\tau_R$", min_value=1.0, max_value=100.0, value=7.5, step=.1, key="tau2")
sigma2 = st.slider(r"$\sigma^\text{cue}$", min_value=.01, max_value=100.0, value=.1, step=.01, key="sigma2")
sigma2r = st.slider(r"$\sigma^\text{cue}$", min_value=.01, max_value=100.0, value=.1, step=.01, key="sigma2r")
tauE2 = st.slider(r"$\tau^\text{cue}_E$", min_value=.01, max_value=100.0, value=5.0, step=.01, key="tauE2")
tauS2 = st.slider(r"$\tau^\text{cue}_S$", min_value=.01, max_value=100.0, value=10.0, step=.01, key="tauS2")
E2r = st.slider(r"$\frac{\tau^\text{rwd}_E}{\tau^\text{cue}_E}$", min_value=1.0, max_value=10000.0, value=10.0, step=1.0, key="E2r")
S2r = st.slider(r"$\frac{\tau^\text{rwd}_S}{\tau^\text{cue}_S}$", min_value=1.0, max_value=10000.0, value=5.0, step=1.0, key="S2r")
b = st.slider(r"$b_\text{cue}$ (baseline for cue neurons)", min_value=.00001, max_value=1.0, value=.04, step=.00001, key="b")
b2 = st.slider(r"$b_\text{rwd}$", min_value=.00001, max_value=100.0, value=.00005, step=.00001, key="b2")

st.line_chart(df2)


st.markdown("# Reward (R)")

tau3 = st.slider(r"$\tau_R$", min_value=1.0, max_value=200.0, value=2.0, step=.1, key="tau3")
sigma3 = st.slider(r"$\sigma$", min_value=.01, max_value=100.0, value=2.6, step=.01, key="sigma3")
tauE3 = st.slider(r"$\tau_E$", min_value=.01, max_value=200.0, value=3.0, step=.01, key="tauE3")
tauS3 = st.slider(r"$\tau_S$", min_value=.01, max_value=200.0, value=6.2, step=.01, key="tauS3")
w1 = st.slider(r"$w^\text{cue-1}$", min_value=.01, max_value=10.0, value=5.8, step=.01, key="w1")
w2 = st.slider(r"$w^\text{cue-2}$", min_value=.01, max_value=10.0, value=3.3, step=.01, key="w2")
w3 = st.slider(r"$w^\text{cue-3}$", min_value=.01, max_value=10.0, value=.001, step=.01, key="w3")
w4 = st.slider(r"$w^\text{cue-4}$", min_value=.01, max_value=10.0, value=2.5, step=.01, key="w4")
scale_rwd = st.slider(r"$w^\text{rwd}$", min_value=.01, max_value=50.0, value=1.1, step=.01, key="scale_rwd")
m = st.slider(r"$m$ (this just scales the entire output response by a constant)", min_value=.5, max_value=10.0, value=5.5, step=.5, key="m")

st.line_chart(df3)







    
    
    
    
