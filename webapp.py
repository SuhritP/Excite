"""
EXCITE - ADHD Detection Through Eye Intelligence
A YC-caliber neural eye tracking analysis platform
"""
import os
import csv
import json
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import threading

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs('uploads', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

state = {'status': 'ready', 'progress': 0, 'message': 'Ready', 'result': None, 'metrics': None}

HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EXCITE | Eye-tracking Classification for Intelligent Therapeutic Evaluation</title>
    <meta name="description" content="EXCITE pioneers non-invasive ADHD detection through AI-powered eye tracking analysis. Screen in minutes, not months.">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #050508;
            --bg-secondary: #0a0a0f;
            --bg-card: rgba(15, 15, 25, 0.6);
            --border-subtle: rgba(255, 255, 255, 0.06);
            --border-glow: rgba(99, 102, 241, 0.4);
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --accent-primary: #818cf8;
            --accent-secondary: #c084fc;
            --accent-tertiary: #22d3ee;
            --success: #34d399;
            --warning: #fbbf24;
            --danger: #f87171;
            --gradient-primary: linear-gradient(135deg, #818cf8 0%, #c084fc 50%, #22d3ee 100%);
            --gradient-glow: linear-gradient(135deg, rgba(129, 140, 248, 0.15) 0%, rgba(192, 132, 252, 0.15) 100%);
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        html {
            scroll-behavior: smooth;
        }

        body {
            font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            overflow-x: hidden;
            line-height: 1.6;
        }

        /* Animated Background */
        .bg-grid {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-image: 
                linear-gradient(rgba(129, 140, 248, 0.03) 1px, transparent 1px),
                linear-gradient(90deg, rgba(129, 140, 248, 0.03) 1px, transparent 1px);
            background-size: 60px 60px;
            pointer-events: none;
            z-index: 0;
        }

        .bg-glow {
            position: fixed;
            width: 600px;
            height: 600px;
            border-radius: 50%;
            filter: blur(120px);
            opacity: 0.15;
            pointer-events: none;
            z-index: 0;
        }

        .bg-glow-1 {
            top: -200px;
            right: -100px;
            background: var(--accent-primary);
            animation: float 20s ease-in-out infinite;
        }

        .bg-glow-2 {
            bottom: -200px;
            left: -100px;
            background: var(--accent-secondary);
            animation: float 25s ease-in-out infinite reverse;
        }

        .bg-glow-3 {
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: var(--accent-tertiary);
            width: 800px;
            height: 800px;
            opacity: 0.08;
            animation: pulse 15s ease-in-out infinite;
        }

        @keyframes float {
            0%, 100% { transform: translate(0, 0) scale(1); }
            33% { transform: translate(30px, -30px) scale(1.05); }
            66% { transform: translate(-20px, 20px) scale(0.95); }
        }

        @keyframes pulse {
            0%, 100% { opacity: 0.08; transform: translate(-50%, -50%) scale(1); }
            50% { opacity: 0.12; transform: translate(-50%, -50%) scale(1.1); }
        }

        /* Navigation */
        nav {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            z-index: 1000;
            padding: 20px 40px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            backdrop-filter: blur(20px);
            background: rgba(5, 5, 8, 0.8);
            border-bottom: 1px solid var(--border-subtle);
        }

        .logo {
            display: flex;
            align-items: center;
            gap: 12px;
            text-decoration: none;
        }

        .logo-icon {
            width: 40px;
            height: 40px;
            background: var(--gradient-primary);
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
        }

        .logo-text {
            font-size: 1.5rem;
            font-weight: 700;
            background: var(--gradient-primary);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            letter-spacing: -0.5px;
        }

        .nav-links {
            display: flex;
            gap: 40px;
            list-style: none;
        }

        .nav-links a {
            color: var(--text-secondary);
            text-decoration: none;
            font-weight: 500;
            font-size: 0.95rem;
            transition: color 0.3s;
        }

        .nav-links a:hover {
            color: var(--text-primary);
        }

        .nav-cta {
            background: var(--gradient-primary);
            color: var(--bg-primary);
            padding: 12px 28px;
            border-radius: 100px;
            font-weight: 600;
            text-decoration: none;
            transition: transform 0.3s, box-shadow 0.3s;
        }

        .nav-cta:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 40px rgba(129, 140, 248, 0.3);
        }

        /* Main Container */
        .container {
            position: relative;
            z-index: 1;
            max-width: 1400px;
            margin: 0 auto;
            padding: 120px 40px 60px;
        }

        /* Hero Section */
        .hero {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 80px;
            align-items: center;
            min-height: calc(100vh - 200px);
            padding: 40px 0;
        }

        .hero-content h1 {
            font-size: 4rem;
            font-weight: 800;
            line-height: 1.1;
            letter-spacing: -2px;
            margin-bottom: 24px;
        }

        .hero-content h1 span {
            background: var(--gradient-primary);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .hero-content p {
            font-size: 1.25rem;
            color: var(--text-secondary);
            margin-bottom: 40px;
            max-width: 500px;
        }

        .hero-stats {
            display: flex;
            gap: 40px;
            margin-top: 60px;
        }

        .stat-item {
            text-align: left;
        }

        .stat-value {
            font-size: 2.5rem;
            font-weight: 700;
            background: var(--gradient-primary);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .stat-label {
            font-size: 0.9rem;
            color: var(--text-muted);
            margin-top: 4px;
        }

        /* Neural Eye Animation */
        .neural-eye-container {
            position: relative;
            width: 100%;
            aspect-ratio: 1;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .neural-eye {
            position: relative;
            width: 400px;
            height: 400px;
        }

        .eye-outer {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 280px;
            height: 280px;
            border: 3px solid rgba(129, 140, 248, 0.3);
            border-radius: 50%;
            animation: rotate 20s linear infinite;
        }

        .eye-outer::before {
            content: '';
            position: absolute;
            top: -8px;
            left: 50%;
            transform: translateX(-50%);
            width: 16px;
            height: 16px;
            background: var(--accent-primary);
            border-radius: 50%;
            box-shadow: 0 0 20px var(--accent-primary);
        }

        .eye-middle {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 200px;
            height: 200px;
            border: 2px solid rgba(192, 132, 252, 0.4);
            border-radius: 50%;
            animation: rotate 15s linear infinite reverse;
        }

        .eye-middle::before {
            content: '';
            position: absolute;
            top: -6px;
            left: 50%;
            transform: translateX(-50%);
            width: 12px;
            height: 12px;
            background: var(--accent-secondary);
            border-radius: 50%;
            box-shadow: 0 0 15px var(--accent-secondary);
        }

        .eye-iris {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 120px;
            height: 120px;
            background: radial-gradient(circle, var(--accent-tertiary) 0%, rgba(34, 211, 238, 0.3) 50%, transparent 70%);
            border-radius: 50%;
            animation: pulse-iris 3s ease-in-out infinite;
        }

        .eye-pupil {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 50px;
            height: 50px;
            background: radial-gradient(circle, #000 0%, #1a1a2e 100%);
            border-radius: 50%;
            box-shadow: 0 0 30px rgba(0, 0, 0, 0.8), inset 0 0 20px rgba(129, 140, 248, 0.3);
        }

        .eye-pupil::after {
            content: '';
            position: absolute;
            top: 8px;
            left: 12px;
            width: 12px;
            height: 12px;
            background: rgba(255, 255, 255, 0.6);
            border-radius: 50%;
        }

        /* Neural Network Lines */
        .neural-lines {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
        }

        .neural-node {
            position: absolute;
            width: 8px;
            height: 8px;
            background: var(--accent-primary);
            border-radius: 50%;
            box-shadow: 0 0 10px var(--accent-primary);
            animation: node-pulse 2s ease-in-out infinite;
        }

        .neural-node:nth-child(1) { top: 10%; left: 20%; animation-delay: 0s; }
        .neural-node:nth-child(2) { top: 25%; left: 5%; animation-delay: 0.3s; }
        .neural-node:nth-child(3) { top: 50%; left: 0%; animation-delay: 0.6s; }
        .neural-node:nth-child(4) { top: 75%; left: 5%; animation-delay: 0.9s; }
        .neural-node:nth-child(5) { top: 90%; left: 20%; animation-delay: 1.2s; }
        .neural-node:nth-child(6) { top: 10%; right: 20%; animation-delay: 0.2s; }
        .neural-node:nth-child(7) { top: 25%; right: 5%; animation-delay: 0.5s; }
        .neural-node:nth-child(8) { top: 50%; right: 0%; animation-delay: 0.8s; }
        .neural-node:nth-child(9) { top: 75%; right: 5%; animation-delay: 1.1s; }
        .neural-node:nth-child(10) { top: 90%; right: 20%; animation-delay: 1.4s; }
        .neural-node:nth-child(11) { top: 0%; left: 50%; animation-delay: 0.4s; }
        .neural-node:nth-child(12) { bottom: 0%; left: 50%; animation-delay: 0.7s; }

        @keyframes rotate {
            from { transform: translate(-50%, -50%) rotate(0deg); }
            to { transform: translate(-50%, -50%) rotate(360deg); }
        }

        @keyframes pulse-iris {
            0%, 100% { transform: translate(-50%, -50%) scale(1); opacity: 1; }
            50% { transform: translate(-50%, -50%) scale(1.1); opacity: 0.8; }
        }

        @keyframes node-pulse {
            0%, 100% { transform: scale(1); opacity: 0.6; }
            50% { transform: scale(1.5); opacity: 1; }
        }

        /* Connecting Lines SVG */
        .neural-svg {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
        }

        .neural-svg line {
            stroke: url(#neural-gradient);
            stroke-width: 1;
            opacity: 0.3;
        }

        /* Upload Section */
        .upload-section {
            margin-top: 40px;
        }

        .section-header {
            text-align: center;
            margin-bottom: 60px;
        }

        .section-tag {
            display: inline-block;
            padding: 8px 20px;
            background: var(--gradient-glow);
            border: 1px solid var(--border-glow);
            border-radius: 100px;
            font-size: 0.85rem;
            font-weight: 500;
            color: var(--accent-primary);
            margin-bottom: 20px;
        }

        .section-header h2 {
            font-size: 3rem;
            font-weight: 700;
            letter-spacing: -1px;
            margin-bottom: 16px;
        }

        .section-header p {
            font-size: 1.1rem;
            color: var(--text-secondary);
            max-width: 600px;
            margin: 0 auto;
        }

        /* Upload Card */
        .upload-card {
            background: var(--bg-card);
            backdrop-filter: blur(40px);
            border: 1px solid var(--border-subtle);
            border-radius: 24px;
            padding: 60px;
            text-align: center;
            position: relative;
            overflow: hidden;
            transition: border-color 0.3s, box-shadow 0.3s;
        }

        .upload-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 1px;
            background: var(--gradient-primary);
            opacity: 0;
            transition: opacity 0.3s;
        }

        .upload-card:hover {
            border-color: var(--border-glow);
            box-shadow: 0 0 60px rgba(129, 140, 248, 0.1);
        }

        .upload-card:hover::before {
            opacity: 1;
        }

        .upload-zone {
            border: 2px dashed var(--border-subtle);
            border-radius: 20px;
            padding: 80px 40px;
            cursor: pointer;
            transition: all 0.3s;
            position: relative;
        }

        .upload-zone:hover {
            border-color: var(--accent-primary);
            background: rgba(129, 140, 248, 0.03);
        }

        .upload-zone.dragover {
            border-color: var(--accent-secondary);
            background: rgba(192, 132, 252, 0.05);
            transform: scale(1.02);
        }

        .upload-icon-wrapper {
            width: 100px;
            height: 100px;
            margin: 0 auto 30px;
            background: var(--gradient-glow);
            border: 1px solid var(--border-glow);
            border-radius: 24px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 40px;
        }

        .upload-title {
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 12px;
        }

        .upload-subtitle {
            color: var(--text-secondary);
            margin-bottom: 30px;
        }

        .file-input {
            display: none;
        }

        .btn-primary {
            background: var(--gradient-primary);
            color: var(--bg-primary);
            border: none;
            padding: 16px 48px;
            font-size: 1rem;
            font-weight: 600;
            font-family: inherit;
            border-radius: 100px;
            cursor: pointer;
            transition: transform 0.3s, box-shadow 0.3s;
        }

        .btn-primary:hover {
            transform: translateY(-3px);
            box-shadow: 0 20px 40px rgba(129, 140, 248, 0.3);
        }

        .btn-primary:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }

        .file-name-display {
            margin-top: 20px;
            font-family: 'JetBrains Mono', monospace;
            color: var(--accent-tertiary);
            font-size: 0.9rem;
        }

        /* Progress Section */
        .progress-section {
            margin-top: 40px;
            padding: 30px;
            background: rgba(129, 140, 248, 0.05);
            border-radius: 16px;
            border: 1px solid var(--border-glow);
        }

        .progress-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
        }

        .progress-title {
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .progress-spinner {
            width: 20px;
            height: 20px;
            border: 2px solid var(--border-subtle);
            border-top-color: var(--accent-primary);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        .progress-percent {
            font-family: 'JetBrains Mono', monospace;
            color: var(--accent-primary);
            font-weight: 600;
        }

        .progress-bar {
            height: 8px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 100px;
            overflow: hidden;
        }

        .progress-fill {
            height: 100%;
            background: var(--gradient-primary);
            border-radius: 100px;
            transition: width 0.3s ease;
            position: relative;
        }

        .progress-fill::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
            animation: shimmer 2s infinite;
        }

        @keyframes shimmer {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }

        .progress-message {
            margin-top: 12px;
            font-size: 0.9rem;
            color: var(--text-secondary);
        }

        /* Preview Card */
        .preview-card {
            background: var(--bg-card);
            backdrop-filter: blur(40px);
            border: 1px solid var(--border-subtle);
            border-radius: 24px;
            padding: 30px;
            margin-top: 40px;
        }

        .preview-card h3 {
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .preview-card h3::before {
            content: '';
            width: 4px;
            height: 20px;
            background: var(--gradient-primary);
            border-radius: 2px;
        }

        .gif-preview {
            width: 100%;
            border-radius: 16px;
            background: var(--bg-secondary);
        }

        /* Results Section */
        .results-section {
            margin-top: 80px;
        }

        /* Result Cards Grid */
        .results-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 24px;
            margin-bottom: 40px;
        }

        .result-card {
            background: var(--bg-card);
            backdrop-filter: blur(40px);
            border: 1px solid var(--border-subtle);
            border-radius: 24px;
            padding: 32px;
            position: relative;
            overflow: hidden;
            transition: transform 0.3s, box-shadow 0.3s;
        }

        .result-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        }

        .result-card.main-result {
            grid-column: span 1;
            text-align: center;
            padding: 48px 32px;
        }

        .result-card h3 {
            font-size: 0.9rem;
            font-weight: 500;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 24px;
        }

        .diagnosis-badge {
            display: inline-block;
            padding: 16px 48px;
            border-radius: 16px;
            font-size: 1.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 2px;
            margin-bottom: 24px;
        }

        .diagnosis-badge.adhd {
            background: linear-gradient(135deg, rgba(248, 113, 113, 0.2), rgba(239, 68, 68, 0.2));
            color: var(--danger);
            border: 1px solid rgba(248, 113, 113, 0.3);
        }

        .diagnosis-badge.control {
            background: linear-gradient(135deg, rgba(52, 211, 153, 0.2), rgba(16, 185, 129, 0.2));
            color: var(--success);
            border: 1px solid rgba(52, 211, 153, 0.3);
        }

        .probability-display {
            margin: 30px 0;
        }

        .probability-circle {
            width: 160px;
            height: 160px;
            margin: 0 auto;
            position: relative;
        }

        .probability-circle canvas {
            position: absolute;
            top: 0;
            left: 0;
        }

        .probability-inner {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            text-align: center;
        }

        .probability-value {
            font-size: 2.5rem;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
        }

        .probability-label {
            font-size: 0.8rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        /* Metrics Grid */
        .metrics-card {
            grid-column: span 2;
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
        }

        .metric-box {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--border-subtle);
            border-radius: 16px;
            padding: 24px;
            text-align: center;
            transition: background 0.3s, border-color 0.3s;
        }

        .metric-box:hover {
            background: rgba(129, 140, 248, 0.05);
            border-color: var(--border-glow);
        }

        .metric-icon {
            width: 48px;
            height: 48px;
            margin: 0 auto 16px;
            background: var(--gradient-glow);
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
        }

        .metric-value {
            font-size: 1.75rem;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
            color: var(--text-primary);
            margin-bottom: 4px;
        }

        .metric-label {
            font-size: 0.85rem;
            color: var(--text-muted);
        }

        /* Charts Grid */
        .charts-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 24px;
            margin-bottom: 40px;
        }

        .chart-card {
            background: var(--bg-card);
            backdrop-filter: blur(40px);
            border: 1px solid var(--border-subtle);
            border-radius: 24px;
            padding: 32px;
        }

        .chart-card h3 {
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 24px;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .chart-card h3::before {
            content: '';
            width: 4px;
            height: 20px;
            background: var(--gradient-primary);
            border-radius: 2px;
        }

        .chart-container {
            position: relative;
            height: 280px;
        }

        /* Full Width Chart */
        .chart-card.full-width {
            grid-column: span 2;
        }

        /* Heatmap Styles */
        .heatmap-container {
            display: grid;
            grid-template-columns: repeat(10, 1fr);
            gap: 4px;
            padding: 20px;
        }

        .heatmap-cell {
            aspect-ratio: 1;
            border-radius: 4px;
            background: var(--accent-primary);
            transition: transform 0.2s;
        }

        .heatmap-cell:hover {
            transform: scale(1.1);
        }

        /* Insights Section */
        .insights-card {
            background: var(--bg-card);
            backdrop-filter: blur(40px);
            border: 1px solid var(--border-subtle);
            border-radius: 24px;
            padding: 32px;
            margin-bottom: 40px;
        }

        .insights-card h3 {
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 24px;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .insights-card h3::before {
            content: '';
            width: 4px;
            height: 20px;
            background: var(--gradient-primary);
            border-radius: 2px;
        }

        .insights-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 16px;
        }

        .insight-item {
            display: flex;
            align-items: flex-start;
            gap: 16px;
            padding: 20px;
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border-subtle);
            border-radius: 16px;
            transition: background 0.3s;
        }

        .insight-item:hover {
            background: rgba(255, 255, 255, 0.04);
        }

        .insight-icon {
            width: 40px;
            height: 40px;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            flex-shrink: 0;
        }

        .insight-icon.success {
            background: rgba(52, 211, 153, 0.15);
            color: var(--success);
        }

        .insight-icon.warning {
            background: rgba(251, 191, 36, 0.15);
            color: var(--warning);
        }

        .insight-icon.info {
            background: rgba(129, 140, 248, 0.15);
            color: var(--accent-primary);
        }

        .insight-icon.danger {
            background: rgba(248, 113, 113, 0.15);
            color: var(--danger);
        }

        .insight-content h4 {
            font-size: 0.95rem;
            font-weight: 600;
            margin-bottom: 4px;
        }

        .insight-content p {
            font-size: 0.85rem;
            color: var(--text-secondary);
            line-height: 1.5;
        }

        /* Clinical Metrics Table */
        .clinical-card {
            background: var(--bg-card);
            backdrop-filter: blur(40px);
            border: 1px solid var(--border-subtle);
            border-radius: 24px;
            padding: 32px;
        }

        .clinical-card h3 {
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 24px;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .clinical-card h3::before {
            content: '';
            width: 4px;
            height: 20px;
            background: var(--gradient-primary);
            border-radius: 2px;
        }

        .clinical-table {
            width: 100%;
            border-collapse: collapse;
        }

        .clinical-table th {
            text-align: left;
            padding: 16px;
            font-size: 0.85rem;
            font-weight: 500;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            border-bottom: 1px solid var(--border-subtle);
        }

        .clinical-table td {
            padding: 16px;
            border-bottom: 1px solid var(--border-subtle);
        }

        .clinical-table tr:last-child td {
            border-bottom: none;
        }

        .clinical-table .metric-name {
            font-weight: 500;
        }

        .clinical-table .metric-val {
            font-family: 'JetBrains Mono', monospace;
            color: var(--accent-primary);
        }

        .clinical-table .status {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 100px;
            font-size: 0.8rem;
            font-weight: 500;
        }

        .status.normal {
            background: rgba(52, 211, 153, 0.15);
            color: var(--success);
        }

        .status.elevated {
            background: rgba(251, 191, 36, 0.15);
            color: var(--warning);
        }

        .status.high {
            background: rgba(248, 113, 113, 0.15);
            color: var(--danger);
        }

        /* Footer */
        footer {
            margin-top: 100px;
            padding: 40px;
            text-align: center;
            border-top: 1px solid var(--border-subtle);
        }

        footer p {
            color: var(--text-muted);
            font-size: 0.9rem;
        }

        /* Animations */
        .fade-in {
            animation: fadeIn 0.6s ease-out;
        }

        .fade-in-up {
            animation: fadeInUp 0.6s ease-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        @keyframes fadeInUp {
            from { 
                opacity: 0; 
                transform: translateY(30px);
            }
            to { 
                opacity: 1; 
                transform: translateY(0);
            }
        }

        .stagger-1 { animation-delay: 0.1s; }
        .stagger-2 { animation-delay: 0.2s; }
        .stagger-3 { animation-delay: 0.3s; }
        .stagger-4 { animation-delay: 0.4s; }

        /* Hidden State */
        .hidden {
            display: none !important;
        }

        /* Content Sections */
        .content-section {
            padding: 120px 0;
            border-top: 1px solid var(--border-subtle);
        }

        /* Features Grid */
        .features-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 24px;
        }

        .feature-card {
            background: var(--bg-card);
            backdrop-filter: blur(40px);
            border: 1px solid var(--border-subtle);
            border-radius: 20px;
            padding: 36px;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }

        .feature-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: var(--gradient-primary);
            opacity: 0;
            transition: opacity 0.3s;
        }

        .feature-card:hover {
            transform: translateY(-8px);
            border-color: var(--border-glow);
            box-shadow: 0 20px 60px rgba(129, 140, 248, 0.15);
        }

        .feature-card:hover::before {
            opacity: 1;
        }

        .feature-icon {
            width: 60px;
            height: 60px;
            background: var(--gradient-glow);
            border: 1px solid var(--border-glow);
            border-radius: 16px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 28px;
            margin-bottom: 24px;
        }

        .feature-card h3 {
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 12px;
            color: var(--text-primary);
        }

        .feature-card p {
            color: var(--text-secondary);
            font-size: 0.95rem;
            line-height: 1.7;
        }

        /* Technology Grid */
        .tech-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 24px;
        }

        .tech-card {
            background: var(--bg-card);
            backdrop-filter: blur(40px);
            border: 1px solid var(--border-subtle);
            border-radius: 20px;
            padding: 36px;
            position: relative;
            transition: all 0.3s ease;
        }

        .tech-card:hover {
            border-color: var(--border-glow);
        }

        .tech-card.large {
            grid-column: span 2;
        }

        .tech-card.full-width {
            grid-column: span 2;
        }

        .tech-card.highlight {
            background: linear-gradient(135deg, rgba(129, 140, 248, 0.1), rgba(192, 132, 252, 0.05));
            border-color: var(--border-glow);
        }

        .tech-number {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--accent-primary);
            margin-bottom: 16px;
            letter-spacing: 1px;
        }

        .tech-card h3 {
            font-size: 1.35rem;
            font-weight: 600;
            margin-bottom: 16px;
            color: var(--text-primary);
        }

        .tech-card p {
            color: var(--text-secondary);
            line-height: 1.7;
        }

        .tech-visual {
            margin-top: 30px;
        }

        .pipeline-flow {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 16px;
            flex-wrap: wrap;
        }

        .pipeline-step {
            background: rgba(129, 140, 248, 0.1);
            border: 1px solid var(--border-glow);
            border-radius: 12px;
            padding: 16px 24px;
            font-weight: 500;
            font-size: 0.9rem;
        }

        .pipeline-arrow {
            color: var(--accent-primary);
            font-size: 1.5rem;
        }

        .biomarkers-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 16px;
            margin-top: 24px;
        }

        .biomarker-item {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 16px;
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--border-subtle);
            border-radius: 12px;
            font-size: 0.9rem;
            color: var(--text-secondary);
        }

        .biomarker-icon {
            font-size: 1.2rem;
        }

        .edge-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            margin-top: 20px;
            padding: 12px 24px;
            background: rgba(52, 211, 153, 0.15);
            border: 1px solid rgba(52, 211, 153, 0.3);
            border-radius: 100px;
            color: var(--success);
            font-weight: 500;
        }

        /* Research Section */
        .research-content {
            display: grid;
            grid-template-columns: 1.5fr 1fr 1fr;
            gap: 24px;
            margin-bottom: 60px;
        }

        .research-card {
            background: var(--bg-card);
            backdrop-filter: blur(40px);
            border: 1px solid var(--border-subtle);
            border-radius: 20px;
            padding: 36px;
        }

        .research-card.main {
            grid-row: span 2;
        }

        .research-card h3 {
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 16px;
            color: var(--text-primary);
        }

        .research-card p {
            color: var(--text-secondary);
            line-height: 1.7;
            margin-bottom: 16px;
        }

        .research-card p:last-child {
            margin-bottom: 0;
        }

        .research-list {
            list-style: none;
            padding: 0;
        }

        .research-list li {
            position: relative;
            padding-left: 24px;
            margin-bottom: 12px;
            color: var(--text-secondary);
            line-height: 1.6;
        }

        .research-list li::before {
            content: '→';
            position: absolute;
            left: 0;
            color: var(--accent-primary);
        }

        .datasets-section h3 {
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 30px;
            text-align: center;
        }

        .dataset-cards {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 24px;
        }

        .dataset-card {
            background: var(--bg-card);
            backdrop-filter: blur(40px);
            border: 1px solid var(--border-subtle);
            border-radius: 20px;
            padding: 32px;
            transition: all 0.3s ease;
        }

        .dataset-card:hover {
            border-color: var(--border-glow);
            transform: translateY(-4px);
        }

        .dataset-icon {
            font-size: 2rem;
            margin-bottom: 16px;
        }

        .dataset-card h4 {
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 8px;
            color: var(--text-primary);
        }

        .dataset-authors {
            font-size: 0.85rem;
            color: var(--text-muted);
            margin-bottom: 12px !important;
        }

        .dataset-desc {
            font-size: 0.9rem;
            color: var(--text-secondary);
            margin-bottom: 16px !important;
        }

        .dataset-link {
            display: inline-flex;
            align-items: center;
            color: var(--accent-primary);
            text-decoration: none;
            font-weight: 500;
            font-size: 0.9rem;
            transition: color 0.3s;
        }

        .dataset-link:hover {
            color: var(--accent-secondary);
        }

        /* About Section */
        .about-content {
            text-align: center;
        }

        .about-main {
            margin-bottom: 48px;
        }

        .about-logo-large {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 20px;
            margin-bottom: 16px;
        }

        .about-eye-icon {
            width: 80px;
            height: 80px;
            background: var(--gradient-primary);
            border-radius: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 40px;
        }

        .about-logo-large span {
            font-size: 3.5rem;
            font-weight: 800;
            background: var(--gradient-primary);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            letter-spacing: -1px;
        }

        .about-tagline {
            font-size: 1.25rem;
            color: var(--text-secondary);
            font-style: italic;
        }

        .about-text {
            max-width: 800px;
            margin: 0 auto 60px;
            text-align: left;
        }

        .about-text p {
            color: var(--text-secondary);
            font-size: 1.1rem;
            line-height: 1.8;
            margin-bottom: 20px;
        }

        .about-lead {
            font-size: 1.25rem !important;
            color: var(--text-primary) !important;
        }

        .about-stats {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 24px;
            max-width: 900px;
            margin: 0 auto;
        }

        .about-stat {
            background: var(--bg-card);
            backdrop-filter: blur(40px);
            border: 1px solid var(--border-subtle);
            border-radius: 20px;
            padding: 32px 24px;
            text-align: center;
        }

        .about-stat-value {
            font-size: 2.5rem;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
            background: var(--gradient-primary);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 8px;
        }

        .about-stat-label {
            font-size: 0.9rem;
            color: var(--text-muted);
        }

        /* Footer Styles */
        footer {
            margin-top: 0;
            border-top: 1px solid var(--border-subtle);
            background: var(--bg-secondary);
        }

        .footer-content {
            max-width: 1400px;
            margin: 0 auto;
            padding: 60px 40px 40px;
            display: grid;
            grid-template-columns: 1fr auto 1fr;
            gap: 60px;
            align-items: start;
        }

        .footer-brand p {
            color: var(--text-muted);
            font-size: 0.9rem;
            margin-top: 12px;
            max-width: 300px;
        }

        .footer-links {
            display: flex;
            gap: 32px;
        }

        .footer-links a {
            color: var(--text-secondary);
            text-decoration: none;
            font-weight: 500;
            transition: color 0.3s;
        }

        .footer-links a:hover {
            color: var(--text-primary);
        }

        .footer-disclaimer {
            text-align: right;
        }

        .footer-disclaimer p {
            font-size: 0.85rem;
            color: var(--text-muted);
            max-width: 350px;
            margin-left: auto;
        }

        .footer-bottom {
            border-top: 1px solid var(--border-subtle);
            padding: 20px 40px;
            text-align: center;
        }

        .footer-bottom p {
            color: var(--text-muted);
            font-size: 0.85rem;
        }

        /* Responsive */
        @media (max-width: 1200px) {
            .hero {
                grid-template-columns: 1fr;
                gap: 60px;
                text-align: center;
            }

            .hero-content p {
                margin-left: auto;
                margin-right: auto;
            }

            .hero-stats {
                justify-content: center;
            }

            .results-grid {
                grid-template-columns: 1fr;
            }

            .metrics-card {
                grid-column: span 1;
            }

            .charts-grid {
                grid-template-columns: 1fr;
            }

            .chart-card.full-width {
                grid-column: span 1;
            }

            .features-grid {
                grid-template-columns: repeat(2, 1fr);
            }

            .tech-grid {
                grid-template-columns: 1fr;
            }

            .tech-card.large,
            .tech-card.full-width {
                grid-column: span 1;
            }

            .biomarkers-grid {
                grid-template-columns: repeat(2, 1fr);
            }

            .research-content {
                grid-template-columns: 1fr;
            }

            .research-card.main {
                grid-row: span 1;
            }

            .about-stats {
                grid-template-columns: repeat(2, 1fr);
            }

            .footer-content {
                grid-template-columns: 1fr;
                gap: 40px;
                text-align: center;
            }

            .footer-brand p {
                margin: 12px auto 0;
            }

            .footer-links {
                justify-content: center;
            }

            .footer-disclaimer {
                text-align: center;
            }

            .footer-disclaimer p {
                margin: 0 auto;
            }
        }

        @media (max-width: 768px) {
            nav {
                padding: 16px 20px;
            }

            .nav-links {
                display: none;
            }

            .container {
                padding: 100px 20px 40px;
            }

            .hero-content h1 {
                font-size: 2.5rem;
            }

            .hero-stats {
                flex-direction: column;
                gap: 24px;
            }

            .neural-eye {
                width: 280px;
                height: 280px;
            }

            .section-header h2 {
                font-size: 2rem;
            }

            .upload-card {
                padding: 30px 20px;
            }

            .upload-zone {
                padding: 40px 20px;
            }

            .metrics-grid {
                grid-template-columns: repeat(2, 1fr);
            }

            .insights-grid {
                grid-template-columns: 1fr;
            }

            .features-grid {
                grid-template-columns: 1fr;
            }

            .biomarkers-grid {
                grid-template-columns: 1fr;
            }

            .pipeline-flow {
                flex-direction: column;
            }

            .pipeline-arrow {
                transform: rotate(90deg);
            }

            .dataset-cards {
                grid-template-columns: 1fr;
            }

            .about-stats {
                grid-template-columns: 1fr;
            }

            .about-logo-large {
                flex-direction: column;
                gap: 12px;
            }

            .about-logo-large span {
                font-size: 2.5rem;
            }

            .content-section {
                padding: 80px 0;
            }

            .footer-links {
                flex-direction: column;
                gap: 16px;
            }
        }

        /* Smooth scroll offset for fixed nav */
        section[id] {
            scroll-margin-top: 100px;
        }
    </style>
</head>
<body>
    <!-- Background Effects -->
    <div class="bg-grid"></div>
    <div class="bg-glow bg-glow-1"></div>
    <div class="bg-glow bg-glow-2"></div>
    <div class="bg-glow bg-glow-3"></div>

    <!-- Navigation -->
    <nav>
        <a href="#" class="logo">
            <div class="logo-icon">👁</div>
            <span class="logo-text">EXCITE</span>
        </a>
        <ul class="nav-links">
            <li><a href="#features">Features</a></li>
            <li><a href="#technology">Technology</a></li>
            <li><a href="#research">Research</a></li>
            <li><a href="#about">About</a></li>
        </ul>
        <a href="#analyze" class="nav-cta">Start Analysis</a>
    </nav>

    <div class="container">
        <!-- Hero Section -->
        <section class="hero">
            <div class="hero-content fade-in-up">
                <h1>Screen for ADHD<br>in <span>Minutes, Not Months</span></h1>
                <p>EXCITE uses AI-powered eye tracking to detect attention disorder patterns through gaze analysis. Non-invasive, portable, and research-backed—making early ADHD screening accessible to schools, clinics, and families worldwide.</p>
                <a href="#analyze" class="btn-primary">Start Screening</a>
                
                <div class="hero-stats">
                    <div class="stat-item">
                        <div class="stat-value">&lt;5 min</div>
                        <div class="stat-label">Screening Time</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">12,000+</div>
                        <div class="stat-label">Training Recordings</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">$50</div>
                        <div class="stat-label">Hardware Cost</div>
                    </div>
                </div>
            </div>

            <div class="neural-eye-container fade-in stagger-2">
                <div class="neural-eye">
                    <svg class="neural-svg" viewBox="0 0 400 400">
                        <defs>
                            <linearGradient id="neural-gradient" x1="0%" y1="0%" x2="100%" y2="100%">
                                <stop offset="0%" style="stop-color:#818cf8;stop-opacity:1" />
                                <stop offset="100%" style="stop-color:#c084fc;stop-opacity:1" />
                            </linearGradient>
                        </defs>
                        <!-- Neural connection lines -->
                        <line x1="80" y1="40" x2="200" y2="200" />
                        <line x1="20" y1="100" x2="200" y2="200" />
                        <line x1="0" y1="200" x2="200" y2="200" />
                        <line x1="20" y1="300" x2="200" y2="200" />
                        <line x1="80" y1="360" x2="200" y2="200" />
                        <line x1="320" y1="40" x2="200" y2="200" />
                        <line x1="380" y1="100" x2="200" y2="200" />
                        <line x1="400" y1="200" x2="200" y2="200" />
                        <line x1="380" y1="300" x2="200" y2="200" />
                        <line x1="320" y1="360" x2="200" y2="200" />
                        <line x1="200" y1="0" x2="200" y2="200" />
                        <line x1="200" y1="400" x2="200" y2="200" />
                    </svg>
                    <div class="neural-lines">
                        <div class="neural-node"></div>
                        <div class="neural-node"></div>
                        <div class="neural-node"></div>
                        <div class="neural-node"></div>
                        <div class="neural-node"></div>
                        <div class="neural-node"></div>
                        <div class="neural-node"></div>
                        <div class="neural-node"></div>
                        <div class="neural-node"></div>
                        <div class="neural-node"></div>
                        <div class="neural-node"></div>
                        <div class="neural-node"></div>
                    </div>
                    <div class="eye-outer"></div>
                    <div class="eye-middle"></div>
                    <div class="eye-iris"></div>
                    <div class="eye-pupil"></div>
                </div>
            </div>
        </section>

        <!-- Upload Section -->
        <section class="upload-section" id="analyze">
            <div class="section-header fade-in-up">
                <span class="section-tag">Begin Screening</span>
                <h2>Upload Eye Tracking Video</h2>
                <p>Watch dots on a screen, and our AI analyzes your eye movements in real-time—extracting saccade patterns, fixation stability, and pupil dynamics to screen for ADHD indicators.</p>
            </div>

            <div class="upload-card fade-in-up stagger-1" id="uploadCard">
                <div class="upload-zone" id="dropZone" onclick="document.getElementById('fileInput').click()">
                    <div class="upload-icon-wrapper">📹</div>
                    <h3 class="upload-title">Drag & drop your video here</h3>
                    <p class="upload-subtitle">or click to browse • MP4, MOV, AVI supported • Max 100MB</p>
                    <input type="file" id="fileInput" class="file-input" accept="video/*">
                    <button class="btn-primary" id="uploadBtn">Select Video File</button>
                    <p class="file-name-display" id="fileName"></p>
                </div>

                <div class="progress-section hidden" id="progressSection">
                    <div class="progress-header">
                        <span class="progress-title">
                            <span class="progress-spinner"></span>
                            Processing
                        </span>
                        <span class="progress-percent" id="progressPercent">0%</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" id="progressFill"></div>
                    </div>
                    <p class="progress-message" id="progressMessage">Initializing neural network...</p>
                </div>
            </div>
        </section>

        <!-- Preview Card -->
        <div class="preview-card hidden" id="previewCard">
            <h3>Eye Tracking Visualization</h3>
            <img class="gif-preview" id="trackedGif" alt="Eye tracking visualization">
        </div>

        <!-- Results Section -->
        <section class="results-section hidden" id="resultsSection">
            <div class="section-header fade-in-up">
                <span class="section-tag">Analysis Complete</span>
                <h2>Diagnostic Results</h2>
                <p>Comprehensive neural analysis of eye movement patterns and attention biomarkers.</p>
            </div>

            <!-- Main Results Grid -->
            <div class="results-grid">
                <div class="result-card main-result fade-in-up">
                    <h3>Diagnosis</h3>
                    <div class="diagnosis-badge" id="diagnosisBadge">--</div>
                    <div class="probability-display">
                        <div class="probability-circle">
                            <canvas id="probCanvas" width="160" height="160"></canvas>
                            <div class="probability-inner">
                                <div class="probability-value" id="probValue">--%</div>
                                <div class="probability-label">Probability</div>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="result-card metrics-card fade-in-up stagger-1">
                    <h3>Key Biomarkers</h3>
                    <div class="metrics-grid">
                        <div class="metric-box">
                            <div class="metric-icon">👁</div>
                            <div class="metric-value" id="metricSaccades">--</div>
                            <div class="metric-label">Saccade Count</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-icon">🎯</div>
                            <div class="metric-value" id="metricFixation">--</div>
                            <div class="metric-label">Avg Fixation (ms)</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-icon">📊</div>
                            <div class="metric-value" id="metricConfidence">--</div>
                            <div class="metric-label">Confidence</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-icon">🔬</div>
                            <div class="metric-value" id="metricPupilVar">--</div>
                            <div class="metric-label">Pupil Variability</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-icon">⏱</div>
                            <div class="metric-value" id="metricFrames">--</div>
                            <div class="metric-label">Frames Analyzed</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-icon">📈</div>
                            <div class="metric-value" id="metricVelocity">--</div>
                            <div class="metric-label">Gaze Velocity</div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Charts Grid -->
            <div class="charts-grid">
                <div class="chart-card fade-in-up">
                    <h3>Gaze Trajectory Analysis</h3>
                    <div class="chart-container">
                        <canvas id="gazeChart"></canvas>
                    </div>
                </div>
                <div class="chart-card fade-in-up stagger-1">
                    <h3>Pupil Diameter Dynamics</h3>
                    <div class="chart-container">
                        <canvas id="pupilChart"></canvas>
                    </div>
                </div>
                <div class="chart-card fade-in-up stagger-2">
                    <h3>Saccade Velocity Profile</h3>
                    <div class="chart-container">
                        <canvas id="velocityChart"></canvas>
                    </div>
                </div>
                <div class="chart-card fade-in-up stagger-3">
                    <h3>Attention Distribution Radar</h3>
                    <div class="chart-container">
                        <canvas id="radarChart"></canvas>
                    </div>
                </div>
            </div>

            <!-- Insights Grid -->
            <div class="insights-card fade-in-up">
                <h3>Clinical Insights</h3>
                <div class="insights-grid" id="insightsGrid">
                    <!-- Populated by JS -->
                </div>
            </div>

            <!-- Clinical Table -->
            <div class="clinical-card fade-in-up">
                <h3>Detailed Biomarker Analysis</h3>
                <table class="clinical-table">
                    <thead>
                        <tr>
                            <th>Biomarker</th>
                            <th>Value</th>
                            <th>Normal Range</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody id="clinicalTableBody">
                        <!-- Populated by JS -->
                    </tbody>
                </table>
            </div>
        </section>
    </div>

        <!-- Features Section -->
        <section class="content-section" id="features">
            <div class="section-header fade-in-up">
                <span class="section-tag">Capabilities</span>
                <h2>Features</h2>
                <p>Revolutionary screening technology that makes ADHD detection accessible, fast, and non-invasive.</p>
            </div>

            <div class="features-grid">
                <div class="feature-card fade-in-up">
                    <div class="feature-icon">🔬</div>
                    <h3>Non-Invasive Screening</h3>
                    <p>No blood tests, no brain scans. Simply watch dots on a screen while our system analyzes your eye movements in real-time.</p>
                </div>
                <div class="feature-card fade-in-up stagger-1">
                    <div class="feature-icon">⚡</div>
                    <h3>Rapid Results</h3>
                    <p>Complete the screening in under 5 minutes. Our AI processes thousands of data points to deliver insights immediately.</p>
                </div>
                <div class="feature-card fade-in-up stagger-2">
                    <div class="feature-icon">📦</div>
                    <h3>Portable Hardware</h3>
                    <p>Our compact Raspberry Pi-based device can be deployed anywhere—classrooms, pediatric offices, or at home.</p>
                </div>
                <div class="feature-card fade-in-up stagger-3">
                    <div class="feature-icon">🧠</div>
                    <h3>Research-Backed Detection</h3>
                    <p>Trained on extensive eye-tracking datasets, our model identifies the subtle gaze patterns that distinguish ADHD from neurotypical attention.</p>
                </div>
                <div class="feature-card fade-in-up stagger-4">
                    <div class="feature-icon">👁</div>
                    <h3>Real-Time Visualization</h3>
                    <p>Watch your gaze patterns in our Eye Tracking Preview interface. See exactly what the AI sees as it analyzes your eye movements.</p>
                </div>
                <div class="feature-card fade-in-up">
                    <div class="feature-icon">🔒</div>
                    <h3>Privacy-First Design</h3>
                    <p>All processing happens locally on the device. Your eye data never leaves the hardware.</p>
                </div>
            </div>
        </section>

        <!-- Technology Section -->
        <section class="content-section" id="technology">
            <div class="section-header fade-in-up">
                <span class="section-tag">Under the Hood</span>
                <h2>Technology</h2>
                <p>State-of-the-art AI architecture powering clinical-grade eye tracking analysis.</p>
            </div>

            <div class="tech-grid">
                <div class="tech-card large fade-in-up">
                    <div class="tech-number">01</div>
                    <h3>Computer Vision Pipeline</h3>
                    <p>Our custom-built camera system captures eye movements at high frequency. Advanced pupil detection algorithms extract gaze coordinates, pupil dilation, and fixation stability—all in real-time.</p>
                    <div class="tech-visual">
                        <div class="pipeline-flow">
                            <div class="pipeline-step">📷 Capture</div>
                            <div class="pipeline-arrow">→</div>
                            <div class="pipeline-step">🔍 Detect</div>
                            <div class="pipeline-arrow">→</div>
                            <div class="pipeline-step">📊 Extract</div>
                            <div class="pipeline-arrow">→</div>
                            <div class="pipeline-step">🧠 Analyze</div>
                        </div>
                    </div>
                </div>

                <div class="tech-card fade-in-up stagger-1">
                    <div class="tech-number">02</div>
                    <h3>Transformer Neural Network</h3>
                    <p>At the heart of EXCITE is a state-of-the-art Transformer encoder, the same architecture powering modern AI breakthroughs. Our model learns temporal patterns in gaze sequences that traditional methods miss.</p>
                </div>

                <div class="tech-card fade-in-up stagger-2">
                    <div class="tech-number">03</div>
                    <h3>Two-Stage Training</h3>
                    <p>We first pre-train on large-scale eye-tracking data (GazeBase: 12,000+ recordings) to learn general gaze dynamics. Then we fine-tune on ADHD-specific datasets to specialize in attention disorder detection.</p>
                </div>

                <div class="tech-card full-width fade-in-up">
                    <div class="tech-number">04</div>
                    <h3>ADHD-Relevant Biomarkers</h3>
                    <p>Our system extracts clinically meaningful features that correlate with attention disorders:</p>
                    <div class="biomarkers-grid">
                        <div class="biomarker-item">
                            <span class="biomarker-icon">↗</span>
                            <span>Saccadic velocity & latency</span>
                        </div>
                        <div class="biomarker-item">
                            <span class="biomarker-icon">◎</span>
                            <span>Fixation stability (BCEA)</span>
                        </div>
                        <div class="biomarker-item">
                            <span class="biomarker-icon">〰</span>
                            <span>Microsaccade frequency</span>
                        </div>
                        <div class="biomarker-item">
                            <span class="biomarker-icon">📈</span>
                            <span>Gaze entropy & predictability</span>
                        </div>
                        <div class="biomarker-item">
                            <span class="biomarker-icon">⬤</span>
                            <span>Pupil diameter dynamics</span>
                        </div>
                        <div class="biomarker-item">
                            <span class="biomarker-icon">🔄</span>
                            <span>Temporal attention patterns</span>
                        </div>
                    </div>
                </div>

                <div class="tech-card highlight fade-in-up">
                    <div class="tech-number">05</div>
                    <h3>Edge Deployment</h3>
                    <p>Optimized to run on Raspberry Pi hardware, bringing lab-grade eye-tracking analysis to a <strong>$50 device</strong>.</p>
                    <div class="edge-badge">
                        <span>🍓</span> Raspberry Pi Compatible
                    </div>
                </div>
            </div>
        </section>

        <!-- Research Section -->
        <section class="content-section" id="research">
            <div class="section-header fade-in-up">
                <span class="section-tag">Scientific Foundation</span>
                <h2>Research</h2>
                <p>Built on peer-reviewed science and validated against clinical diagnoses.</p>
            </div>

            <div class="research-content">
                <div class="research-card main fade-in-up">
                    <h3>The Science of Gaze</h3>
                    <p>Eye movements are controlled by the same neural circuits involved in attention and executive function—the core deficits in ADHD. Research shows that individuals with ADHD exhibit distinct patterns:</p>
                    <ul class="research-list">
                        <li>More frequent saccades during sustained attention tasks</li>
                        <li>Reduced fixation stability and increased drift</li>
                        <li>Altered pupil responses to cognitive load</li>
                        <li>Different temporal patterns in gaze sequences</li>
                    </ul>
                </div>

                <div class="research-card fade-in-up stagger-1">
                    <h3>Our Approach</h3>
                    <p>EXCITE leverages transfer learning from the GazeBase dataset, one of the largest eye-tracking repositories with recordings from <strong>322 participants</strong> across multiple sessions. By pre-training on this data, our model learns robust representations of eye movement dynamics before specializing on ADHD detection.</p>
                </div>

                <div class="research-card fade-in-up stagger-2">
                    <h3>Validation</h3>
                    <p>Our model is validated against clinical ADHD diagnoses, with ongoing studies to establish sensitivity and specificity benchmarks. EXCITE is a screening tool designed to <strong>complement—not replace</strong>—professional clinical evaluation.</p>
                </div>
            </div>

            <div class="datasets-section fade-in-up">
                <h3>Datasets</h3>
                <div class="dataset-cards">
                    <div class="dataset-card">
                        <div class="dataset-icon">📊</div>
                        <h4>GazeBase Data Repository</h4>
                        <p class="dataset-authors">Griffith, H., Lohr, D., Abdulin, E., & Komogortsev, O. (2020)</p>
                        <p class="dataset-desc">Large-scale, multi-stimulus, longitudinal eye movement dataset.</p>
                        <a href="https://doi.org/10.6084/m9.figshare.12912257" target="_blank" class="dataset-link">
                            View on figshare →
                        </a>
                    </div>
                    <div class="dataset-card">
                        <div class="dataset-icon">🔬</div>
                        <h4>ADHD Pupil Size Dataset</h4>
                        <p class="dataset-authors">Krejtz, K., et al. (2018)</p>
                        <p class="dataset-desc">Pupil size dataset specifically collected for ADHD research.</p>
                        <a href="https://doi.org/10.6084/m9.figshare.7218725" target="_blank" class="dataset-link">
                            View on figshare →
                        </a>
                    </div>
                </div>
            </div>
        </section>

        <!-- About Section -->
        <section class="content-section" id="about">
            <div class="section-header fade-in-up">
                <span class="section-tag">Our Mission</span>
                <h2>About EXCITE</h2>
            </div>

            <div class="about-content fade-in-up">
                <div class="about-main">
                    <div class="about-logo-large">
                        <div class="about-eye-icon">👁</div>
                        <span>EXCITE</span>
                    </div>
                    <p class="about-tagline">Eye-tracking Classification for Intelligent Therapeutic Evaluation</p>
                </div>

                <div class="about-text">
                    <p class="about-lead">EXCITE is pioneering a new frontier in ADHD detection. Our mission is to develop an accessible, non-invasive screening tool that can identify ADHD indicators through eye movement analysis.</p>
                    
                    <p>Traditional ADHD diagnosis relies on subjective behavioral assessments and lengthy clinical evaluations. EXCITE offers a different approach—<strong>using the eyes as a window into cognitive function</strong>. By analyzing how individuals track visual stimuli, our system can detect patterns associated with attention disorders in minutes, not months.</p>
                    
                    <p>Built by researchers passionate about bridging neuroscience and technology, EXCITE aims to make early ADHD screening available to schools, clinics, and families worldwide.</p>
                </div>

                <div class="about-stats">
                    <div class="about-stat">
                        <div class="about-stat-value">12,000+</div>
                        <div class="about-stat-label">Training Recordings</div>
                    </div>
                    <div class="about-stat">
                        <div class="about-stat-value">&lt;5 min</div>
                        <div class="about-stat-label">Screening Time</div>
                    </div>
                    <div class="about-stat">
                        <div class="about-stat-value">$50</div>
                        <div class="about-stat-label">Hardware Cost</div>
                    </div>
                    <div class="about-stat">
                        <div class="about-stat-value">100%</div>
                        <div class="about-stat-label">Local Processing</div>
                    </div>
                </div>
            </div>
        </section>
    </div>

    <!-- Footer -->
    <footer>
        <div class="footer-content">
            <div class="footer-brand">
                <div class="logo">
                    <div class="logo-icon">👁</div>
                    <span class="logo-text">EXCITE</span>
                </div>
                <p>Eye-tracking Classification for Intelligent Therapeutic Evaluation</p>
            </div>
            <div class="footer-links">
                <a href="#features">Features</a>
                <a href="#technology">Technology</a>
                <a href="#research">Research</a>
                <a href="#about">About</a>
            </div>
            <div class="footer-disclaimer">
                <p><strong>Disclaimer:</strong> EXCITE is a screening tool designed to complement—not replace—professional clinical evaluation for ADHD.</p>
            </div>
        </div>
        <div class="footer-bottom">
            <p>© 2026 EXCITE — Built for the future of cognitive diagnostics</p>
        </div>
    </footer>

    <script>
        // Chart instances
        let gazeChart, pupilChart, velocityChart, radarChart;

        // DOM Elements
        const dropZone = document.getElementById('dropZone');
        const fileInput = document.getElementById('fileInput');
        const progressSection = document.getElementById('progressSection');
        const previewCard = document.getElementById('previewCard');
        const resultsSection = document.getElementById('resultsSection');

        // Drag and drop
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });

        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('dragover');
        });

        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            if (e.dataTransfer.files.length) {
                fileInput.files = e.dataTransfer.files;
                handleFile(e.dataTransfer.files[0]);
            }
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length) {
                handleFile(e.target.files[0]);
            }
        });

        function handleFile(file) {
            document.getElementById('fileName').textContent = '📎 ' + file.name;
            upload(file);
        }

        function resetUI() {
            previewCard.classList.add('hidden');
            resultsSection.classList.add('hidden');
            document.getElementById('trackedGif').src = '';
            document.getElementById('progressFill').style.width = '0%';
            if (gazeChart) gazeChart.destroy();
            if (pupilChart) pupilChart.destroy();
            if (velocityChart) velocityChart.destroy();
            if (radarChart) radarChart.destroy();
        }

        function upload(file) {
            resetUI();

            const fd = new FormData();
            fd.append('video', file);

            progressSection.classList.remove('hidden');
            document.getElementById('progressMessage').textContent = 'Uploading video to neural network...';
            document.getElementById('progressPercent').textContent = '0%';

            fetch('/upload', { method: 'POST', body: fd })
                .then(r => r.json())
                .then(d => {
                    if (d.success) poll();
                    else {
                        document.getElementById('progressMessage').textContent = '❌ Error: ' + d.error;
                    }
                });
        }

        function poll() {
            fetch('/status')
                .then(r => r.json())
                .then(d => {
                    document.getElementById('progressMessage').textContent = d.message;
                    document.getElementById('progressFill').style.width = d.progress + '%';
                    document.getElementById('progressPercent').textContent = d.progress + '%';

                    if (d.status === 'done') {
                        showResults(d.result, d.metrics);
                    } else if (d.status === 'error') {
                        document.getElementById('progressMessage').textContent = '❌ Error: ' + d.message;
                    } else {
                        setTimeout(poll, 400);
                    }
                });
        }

        function showResults(result, metrics) {
            progressSection.classList.add('hidden');
            previewCard.classList.remove('hidden');
            resultsSection.classList.remove('hidden');

            // Scroll to results
            resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

            // Load GIF
            document.getElementById('trackedGif').src = '/gif?t=' + Date.now();

            // Diagnosis badge
            const badge = document.getElementById('diagnosisBadge');
            badge.textContent = result.prediction;
            badge.className = 'diagnosis-badge ' + (result.prediction === 'ADHD' ? 'adhd' : 'control');

            // Probability
            document.getElementById('probValue').textContent = result.probability + '%';
            drawProbabilityRing(result.probability);

            // Metrics
            document.getElementById('metricSaccades').textContent = metrics ? metrics.saccade_count : '--';
            document.getElementById('metricFixation').textContent = metrics ? metrics.avg_fixation : '--';
            document.getElementById('metricConfidence').textContent = result.confidence + '%';
            document.getElementById('metricPupilVar').textContent = metrics ? metrics.pupil_std : '--';
            document.getElementById('metricFrames').textContent = result.frames;
            document.getElementById('metricVelocity').textContent = metrics && metrics.avg_velocity ? metrics.avg_velocity.toFixed(1) + '°/s' : '--';

            // Charts
            if (metrics) {
                if (metrics.gaze_x && metrics.gaze_y) {
                    drawGazeChart(metrics.gaze_x, metrics.gaze_y);
                }
                if (metrics.pupil_data) {
                    drawPupilChart(metrics.pupil_data);
                }
                if (metrics.velocities) {
                    drawVelocityChart(metrics.velocities);
                }
                drawRadarChart(result, metrics);
            }

            // Insights
            generateInsights(result, metrics);

            // Clinical table
            generateClinicalTable(result, metrics);
        }

        function drawProbabilityRing(prob) {
            const canvas = document.getElementById('probCanvas');
            const ctx = canvas.getContext('2d');
            const centerX = 80, centerY = 80, radius = 65;

            ctx.clearRect(0, 0, 160, 160);

            // Background ring
            ctx.beginPath();
            ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
            ctx.strokeStyle = 'rgba(255,255,255,0.1)';
            ctx.lineWidth = 10;
            ctx.stroke();

            // Progress ring
            const endAngle = (prob / 100) * 2 * Math.PI - Math.PI / 2;
            ctx.beginPath();
            ctx.arc(centerX, centerY, radius, -Math.PI / 2, endAngle);

            const gradient = ctx.createLinearGradient(0, 0, 160, 160);
            if (prob > 50) {
                gradient.addColorStop(0, '#f87171');
                gradient.addColorStop(1, '#ef4444');
            } else {
                gradient.addColorStop(0, '#34d399');
                gradient.addColorStop(1, '#22d3ee');
            }

            ctx.strokeStyle = gradient;
            ctx.lineWidth = 10;
            ctx.lineCap = 'round';
            ctx.stroke();
        }

        function drawGazeChart(gazeX, gazeY) {
            const ctx = document.getElementById('gazeChart').getContext('2d');
            if (gazeChart) gazeChart.destroy();

            const labels = gazeX.map((_, i) => i);

            gazeChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Horizontal (X)',
                        data: gazeX,
                        borderColor: '#818cf8',
                        backgroundColor: 'rgba(129, 140, 248, 0.1)',
                        tension: 0.4,
                        fill: true,
                        pointRadius: 0,
                        borderWidth: 2
                    }, {
                        label: 'Vertical (Y)',
                        data: gazeY,
                        borderColor: '#22d3ee',
                        backgroundColor: 'rgba(34, 211, 238, 0.1)',
                        tension: 0.4,
                        fill: true,
                        pointRadius: 0,
                        borderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: { intersect: false, mode: 'index' },
                    plugins: {
                        legend: {
                            labels: { color: '#94a3b8', usePointStyle: true, pointStyle: 'circle' }
                        }
                    },
                    scales: {
                        x: { display: false },
                        y: {
                            grid: { color: 'rgba(255,255,255,0.05)', drawBorder: false },
                            ticks: { color: '#64748b' }
                        }
                    }
                }
            });
        }

        function drawPupilChart(pupilData) {
            const ctx = document.getElementById('pupilChart').getContext('2d');
            if (pupilChart) pupilChart.destroy();

            const labels = pupilData.map((_, i) => i);

            pupilChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Pupil Diameter (px)',
                        data: pupilData,
                        borderColor: '#c084fc',
                        backgroundColor: 'rgba(192, 132, 252, 0.15)',
                        tension: 0.4,
                        fill: true,
                        pointRadius: 0,
                        borderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: { intersect: false, mode: 'index' },
                    plugins: {
                        legend: {
                            labels: { color: '#94a3b8', usePointStyle: true, pointStyle: 'circle' }
                        }
                    },
                    scales: {
                        x: { display: false },
                        y: {
                            grid: { color: 'rgba(255,255,255,0.05)', drawBorder: false },
                            ticks: { color: '#64748b' }
                        }
                    }
                }
            });
        }

        function drawVelocityChart(velocities) {
            const ctx = document.getElementById('velocityChart').getContext('2d');
            if (velocityChart) velocityChart.destroy();

            const labels = velocities.map((_, i) => i);

            velocityChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Saccade Velocity (°/s)',
                        data: velocities,
                        backgroundColor: velocities.map(v => v > 30 ? 'rgba(248, 113, 113, 0.6)' : 'rgba(129, 140, 248, 0.6)'),
                        borderColor: velocities.map(v => v > 30 ? '#f87171' : '#818cf8'),
                        borderWidth: 1,
                        borderRadius: 4,
                        barPercentage: 0.8
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            labels: { color: '#94a3b8', usePointStyle: true, pointStyle: 'rect' }
                        }
                    },
                    scales: {
                        x: { display: false },
                        y: {
                            grid: { color: 'rgba(255,255,255,0.05)', drawBorder: false },
                            ticks: { color: '#64748b' }
                        }
                    }
                }
            });
        }

        function drawRadarChart(result, metrics) {
            const ctx = document.getElementById('radarChart').getContext('2d');
            if (radarChart) radarChart.destroy();

            // Normalize values to 0-100 scale
            const saccadeScore = Math.min(100, (metrics.saccade_count / 100) * 100);
            const fixationScore = Math.min(100, (metrics.avg_fixation / 50) * 100);
            const pupilScore = Math.min(100, (metrics.pupil_std / 10) * 100);
            const velocityScore = metrics.avg_velocity ? Math.min(100, (metrics.avg_velocity / 50) * 100) : 50;
            const confidenceScore = result.confidence;

            radarChart = new Chart(ctx, {
                type: 'radar',
                data: {
                    labels: ['Saccade Freq.', 'Fixation Stability', 'Pupil Variability', 'Gaze Velocity', 'Confidence'],
                    datasets: [{
                        label: 'Patient Profile',
                        data: [saccadeScore, fixationScore, pupilScore, velocityScore, confidenceScore],
                        backgroundColor: 'rgba(129, 140, 248, 0.2)',
                        borderColor: '#818cf8',
                        borderWidth: 2,
                        pointBackgroundColor: '#818cf8',
                        pointBorderColor: '#fff',
                        pointRadius: 4
                    }, {
                        label: 'Typical Range',
                        data: [40, 60, 30, 35, 70],
                        backgroundColor: 'rgba(34, 211, 238, 0.1)',
                        borderColor: '#22d3ee',
                        borderWidth: 1,
                        borderDash: [5, 5],
                        pointRadius: 0
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            labels: { color: '#94a3b8', usePointStyle: true, pointStyle: 'circle' }
                        }
                    },
                    scales: {
                        r: {
                            angleLines: { color: 'rgba(255,255,255,0.1)' },
                            grid: { color: 'rgba(255,255,255,0.05)' },
                            pointLabels: { color: '#94a3b8', font: { size: 11 } },
                            ticks: { display: false },
                            suggestedMin: 0,
                            suggestedMax: 100
                        }
                    }
                }
            });
        }

        function generateInsights(result, metrics) {
            const grid = document.getElementById('insightsGrid');
            grid.innerHTML = '';

            const insights = [];

            if (result.prediction === 'ADHD') {
                insights.push({
                    icon: 'danger',
                    iconText: '⚠',
                    title: 'Elevated Attention Variability',
                    desc: 'Gaze patterns show higher than typical variability, consistent with ADHD attention profiles.'
                });
            } else {
                insights.push({
                    icon: 'success',
                    iconText: '✓',
                    title: 'Typical Attention Patterns',
                    desc: 'Eye movement patterns fall within the expected range for neurotypical attention profiles.'
                });
            }

            if (result.confidence >= 70) {
                insights.push({
                    icon: 'success',
                    iconText: '✓',
                    title: 'High Confidence Classification',
                    desc: `Model confidence of ${result.confidence}% indicates strong pattern recognition.`
                });
            } else if (result.confidence >= 40) {
                insights.push({
                    icon: 'warning',
                    iconText: '!',
                    title: 'Moderate Confidence',
                    desc: 'Consider a longer video sample for improved accuracy and confidence levels.'
                });
            } else {
                insights.push({
                    icon: 'info',
                    iconText: 'i',
                    title: 'Borderline Classification',
                    desc: 'Results are near the decision boundary. Clinical consultation recommended.'
                });
            }

            if (metrics) {
                if (metrics.saccade_count > 60) {
                    insights.push({
                        icon: 'warning',
                        iconText: '!',
                        title: 'Elevated Saccade Frequency',
                        desc: `Detected ${metrics.saccade_count} saccades, above typical threshold of 50.`
                    });
                }

                if (metrics.avg_fixation < 20) {
                    insights.push({
                        icon: 'danger',
                        iconText: '⚠',
                        title: 'Reduced Fixation Duration',
                        desc: 'Short fixation periods may indicate difficulty maintaining visual attention.'
                    });
                }

                insights.push({
                    icon: 'info',
                    iconText: 'i',
                    title: 'Comprehensive Analysis',
                    desc: `Analyzed ${result.frames} frames with ${metrics.saccade_count} detected saccades.`
                });
            }

            insights.forEach(item => {
                const div = document.createElement('div');
                div.className = 'insight-item';
                div.innerHTML = `
                    <div class="insight-icon ${item.icon}">${item.iconText}</div>
                    <div class="insight-content">
                        <h4>${item.title}</h4>
                        <p>${item.desc}</p>
                    </div>
                `;
                grid.appendChild(div);
            });
        }

        function generateClinicalTable(result, metrics) {
            const tbody = document.getElementById('clinicalTableBody');
            tbody.innerHTML = '';

            const rows = [
                {
                    name: 'Saccade Count',
                    value: metrics ? metrics.saccade_count : '--',
                    range: '20-50',
                    status: metrics && metrics.saccade_count > 50 ? 'elevated' : 'normal'
                },
                {
                    name: 'Average Fixation Duration',
                    value: metrics ? metrics.avg_fixation + ' ms' : '--',
                    range: '150-400 ms',
                    status: metrics && metrics.avg_fixation < 150 ? 'elevated' : 'normal'
                },
                {
                    name: 'Pupil Diameter Std Dev',
                    value: metrics ? metrics.pupil_std : '--',
                    range: '< 5.0',
                    status: metrics && metrics.pupil_std > 5 ? 'elevated' : 'normal'
                },
                {
                    name: 'Average Gaze Velocity',
                    value: metrics && metrics.avg_velocity ? metrics.avg_velocity.toFixed(2) + ' °/s' : '--',
                    range: '< 30 °/s',
                    status: metrics && metrics.avg_velocity > 30 ? 'elevated' : 'normal'
                },
                {
                    name: 'Peak Velocity',
                    value: metrics && metrics.peak_velocity ? metrics.peak_velocity.toFixed(2) + ' °/s' : '--',
                    range: '< 100 °/s',
                    status: metrics && metrics.peak_velocity > 100 ? 'high' : 'normal'
                },
                {
                    name: 'Gaze Entropy',
                    value: metrics && metrics.gaze_entropy ? metrics.gaze_entropy.toFixed(3) : '--',
                    range: '< 0.7',
                    status: metrics && metrics.gaze_entropy > 0.7 ? 'elevated' : 'normal'
                },
                {
                    name: 'ADHD Probability',
                    value: result.probability + '%',
                    range: '< 50%',
                    status: result.probability > 70 ? 'high' : result.probability > 50 ? 'elevated' : 'normal'
                },
                {
                    name: 'Classification Confidence',
                    value: result.confidence + '%',
                    range: '> 60%',
                    status: result.confidence >= 60 ? 'normal' : 'elevated'
                }
            ];

            rows.forEach(row => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td class="metric-name">${row.name}</td>
                    <td class="metric-val">${row.value}</td>
                    <td>${row.range}</td>
                    <td><span class="status ${row.status}">${row.status.charAt(0).toUpperCase() + row.status.slice(1)}</span></td>
                `;
                tbody.appendChild(tr);
            });
        }

        // Smooth scroll for nav links
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }
            });
        });
    </script>
</body>
</html>'''

@app.route('/')
def index():
    return HTML_TEMPLATE

@app.route('/upload', methods=['POST'])
def upload():
    global state
    if 'video' not in request.files:
        return jsonify({'success': False, 'error': 'No video'})
    f = request.files['video']
    path = os.path.join('uploads', secure_filename(f.filename))
    f.save(path)
    state = {'status': 'processing', 'progress': 0, 'message': 'Starting neural analysis...', 'result': None, 'metrics': None}
    threading.Thread(target=process, args=(path,)).start()
    return jsonify({'success': True})

@app.route('/status')
def status():
    return jsonify(state)

@app.route('/gif')
def serve_gif():
    return send_file('outputs/tracked.gif', mimetype='image/gif')

def process(video_path):
    global state
    import cv2
    from eyetrack import process_frame, crop_to_aspect_ratio
    
    try:
        state['message'] = 'Loading video into memory...'
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 100
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        
        tracked_frames = []
        results = []
        gaze_x_list = []
        gaze_y_list = []
        pupil_list = []
        frame_id = 0
        
        state['message'] = 'Extracting eye features with neural network...'
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            state['progress'] = int(frame_id / total * 70)
            state['message'] = f'Neural tracking: Frame {frame_id}/{total}'
            
            try:
                result = process_frame(frame.copy())
                display_frame = crop_to_aspect_ratio(frame, 320, 240)
                
                if result is not None:
                    center = result[0] if result[0] else (160, 120)
                    size = result[1] if result[1] else (30, 30)
                    
                    px, py = int(center[0] * 320/640), int(center[1] * 240/480)
                    pupil_d = (size[0] + size[1]) / 2 * 320/640
                    
                    cv2.ellipse(display_frame, (px, py), (int(pupil_d/2), int(pupil_d/2)), 
                               0, 0, 360, (0, 255, 0), 2)
                    cv2.circle(display_frame, (px, py), 3, (0, 0, 255), -1)
                    valid = 0
                else:
                    px, py, pupil_d = 160, 120, 30
                    valid = 1
                    
            except Exception as e:
                px, py, pupil_d = 160, 120, 30
                valid = 1
                display_frame = cv2.resize(frame, (320, 240))
            
            cv2.putText(display_frame, f'Frame: {frame_id}', (10, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            if frame_id % 3 == 0:
                tracked_frames.append(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
            
            theta_x = (px - 160) / 10
            theta_y = (py - 120) / 10
            
            gaze_x_list.append(theta_x)
            gaze_y_list.append(theta_y)
            pupil_list.append(pupil_d)
            
            results.append([frame_id, px, py, pupil_d, theta_x, theta_y, valid])
            frame_id += 1
        
        cap.release()
        
        # Save as GIF
        state['message'] = 'Generating visualization preview...'
        state['progress'] = 75
        from PIL import Image
        if tracked_frames:
            imgs = [Image.fromarray(f) for f in tracked_frames[:100]]
            imgs[0].save('outputs/tracked.gif', save_all=True, append_images=imgs[1:], 
                        duration=100, loop=0)
        
        # Calculate advanced metrics
        state['message'] = 'Computing attention biomarkers...'
        state['progress'] = 80
        
        # Saccade detection (velocity threshold)
        velocities = []
        for i in range(1, len(gaze_x_list)):
            dx = gaze_x_list[i] - gaze_x_list[i-1]
            dy = gaze_y_list[i] - gaze_y_list[i-1]
            v = np.sqrt(dx**2 + dy**2) * fps
            velocities.append(v)
        
        saccade_threshold = 30  # degrees per second
        saccade_count = sum(1 for v in velocities if v > saccade_threshold)
        
        # Average fixation duration
        fixation_frames = sum(1 for v in velocities if v < saccade_threshold)
        avg_fixation = int((fixation_frames / max(len(velocities), 1)) * (1000 / fps))
        
        # Pupil stats
        pupil_std = round(np.std(pupil_list), 2) if pupil_list else 0
        
        # Velocity stats
        avg_velocity = np.mean(velocities) if velocities else 0
        peak_velocity = np.max(velocities) if velocities else 0
        
        # Gaze entropy (measure of randomness in gaze)
        gaze_x_bins = np.histogram(gaze_x_list, bins=10)[0]
        gaze_x_bins = gaze_x_bins / max(np.sum(gaze_x_bins), 1)
        gaze_entropy = -np.sum(gaze_x_bins * np.log2(gaze_x_bins + 1e-10))
        
        # Subsample for charts
        step = max(1, len(gaze_x_list) // 200)
        vel_step = max(1, len(velocities) // 100)
        
        metrics = {
            'saccade_count': saccade_count,
            'avg_fixation': avg_fixation,
            'pupil_std': pupil_std,
            'avg_velocity': round(avg_velocity, 2),
            'peak_velocity': round(peak_velocity, 2),
            'gaze_entropy': round(gaze_entropy, 3),
            'gaze_x': gaze_x_list[::step],
            'gaze_y': gaze_y_list[::step],
            'pupil_data': pupil_list[::step],
            'velocities': velocities[::vel_step]
        }
        
        # Run ML prediction
        state['message'] = 'Running ADHD classification model...'
        state['progress'] = 90
        
        from inference import ADHDDetector
        detector = ADHDDetector()
        
        preds = []
        for r in results:
            res = detector.add_frame(r[4], r[5], r[3], r[6])
            if res:
                preds.append(res['probability'])
        
        state['progress'] = 100
        
        if preds:
            avg = np.mean(preds)
            state['result'] = {
                'prediction': 'ADHD' if avg > 0.5 else 'Control',
                'probability': round(avg * 100, 1),
                'confidence': round(abs(avg - 0.5) * 200, 1),
                'frames': len(results)
            }
        else:
            state['result'] = {
                'prediction': 'Need more data',
                'probability': 0,
                'confidence': 0,
                'frames': len(results)
            }
        
        state['metrics'] = metrics
        state['status'] = 'done'
        state['message'] = 'Neural analysis complete!'
        
    except Exception as e:
        state['status'] = 'error'
        state['message'] = str(e)

if __name__ == '__main__':
    print()
    print('╔══════════════════════════════════════════════════════════════╗')
    print('║                                                              ║')
    print('║   ███████╗██╗  ██╗ ██████╗██╗████████╗███████╗               ║')
    print('║   ██╔════╝╚██╗██╔╝██╔════╝██║╚══██╔══╝██╔════╝               ║')
    print('║   █████╗   ╚███╔╝ ██║     ██║   ██║   █████╗                 ║')
    print('║   ██╔══╝   ██╔██╗ ██║     ██║   ██║   ██╔══╝                 ║')
    print('║   ███████╗██╔╝ ██╗╚██████╗██║   ██║   ███████╗               ║')
    print('║   ╚══════╝╚═╝  ╚═╝ ╚═════╝╚═╝   ╚═╝   ╚══════╝               ║')
    print('║                                                              ║')
    print('║   Neural Eye Intelligence for ADHD Detection                 ║')
    print('║                                                              ║')
    print('╠══════════════════════════════════════════════════════════════╣')
    print('║   🌐 Open: http://localhost:5000                             ║')
    print('╚══════════════════════════════════════════════════════════════╝')
    print()
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
