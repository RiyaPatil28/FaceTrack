#!/usr/bin/env python3
"""
Mobile Integration Feature for Smart Attendance System
Adds mobile API routes to the main Streamlit app
"""

import streamlit as st
import requests
import json
from datetime import datetime, date

def mobile_integration_page(db):
    """Mobile Integration and API Testing Page"""
    st.header("📱 Mobile App Integration")
    st.markdown("Test the mobile API and view mobile app features")
    
    # API Status Check
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔗 API Status")
        if st.button("Check API Connection"):
            try:
                response = requests.get("http://localhost:3000/api/status", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    st.success("✅ Mobile API is running!")
                    st.json(data)
                else:
                    st.error(f"❌ API returned status code: {response.status_code}")
            except requests.exceptions.ConnectionError:
                st.error("❌ Mobile API is not running on port 3000")
                st.info("The mobile API should be started with: `python mobile_api.py`")
            except Exception as e:
                st.error(f"❌ Error connecting to API: {e}")
    
    with col2:
        st.subheader("📊 Mobile Dashboard Preview")
        if st.button("Load Mobile Dashboard"):
            try:
                response = requests.get("http://localhost:3000/api/dashboard/stats", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    if data['success']:
                        stats = data['data']
                        
                        # Display stats in mobile-like format
                        st.markdown("### Today's Stats")
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Total Employees", stats['today']['total_employees'])
                        with col_b:
                            st.metric("Present Today", stats['today']['present'])
                        with col_c:
                            st.metric("Attendance Rate", f"{stats['today']['attendance_rate']}%")
                        
                        # Recent attendance
                        if stats['recent_attendance']:
                            st.markdown("### Recent Check-ins")
                            for item in stats['recent_attendance']:
                                with st.container():
                                    col_name, col_time, col_conf = st.columns([3, 2, 1])
                                    with col_name:
                                        st.write(f"👤 **{item['name']}**")
                                        st.caption(item['department'])
                                    with col_time:
                                        st.write(item['check_in_time'])
                                    with col_conf:
                                        st.success(f"{item['confidence']}%")
                else:
                    st.error("Failed to load mobile dashboard data")
            except Exception as e:
                st.error(f"Error loading dashboard: {e}")
    
    st.markdown("---")
    
    # API Endpoints Documentation
    st.subheader("📋 Available API Endpoints")
    
    endpoints = [
        {
            "method": "GET",
            "endpoint": "/api/status",
            "description": "Check API health status",
            "example": "http://localhost:3000/api/status"
        },
        {
            "method": "GET", 
            "endpoint": "/api/employees",
            "description": "Get all employees list",
            "example": "http://localhost:3000/api/employees"
        },
        {
            "method": "GET",
            "endpoint": "/api/attendance/today", 
            "description": "Get today's attendance statistics",
            "example": "http://localhost:3000/api/attendance/today"
        },
        {
            "method": "GET",
            "endpoint": "/api/attendance/range",
            "description": "Get attendance for date range",
            "example": "http://localhost:3000/api/attendance/range?start_date=2025-01-28&end_date=2025-01-28"
        },
        {
            "method": "GET",
            "endpoint": "/api/dashboard/stats",
            "description": "Get comprehensive dashboard statistics",
            "example": "http://localhost:3000/api/dashboard/stats"
        },
        {
            "method": "GET",
            "endpoint": "/api/employee/{id}/history",
            "description": "Get attendance history for specific employee",
            "example": "http://localhost:3000/api/employee/CELEB001/history?days=30"
        }
    ]
    
    for endpoint in endpoints:
        with st.expander(f"{endpoint['method']} {endpoint['endpoint']}"):
            st.write(endpoint['description'])
            st.code(endpoint['example'])
            
            if st.button(f"Test {endpoint['endpoint']}", key=endpoint['endpoint']):
                try:
                    response = requests.get(endpoint['example'].replace('localhost', '127.0.0.1'), timeout=5)
                    if response.status_code == 200:
                        st.success("✅ Success")
                        st.json(response.json())
                    else:
                        st.error(f"❌ Status: {response.status_code}")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
    
    st.markdown("---")
    
    # Mobile App Demo
    st.subheader("📱 Mobile App Demo")
    st.markdown("Preview of the mobile app interface:")
    
    if st.button("Open Mobile App Demo"):
        st.markdown("""
        **Mobile App Features:**
        - 📊 Real-time attendance dashboard
        - 👥 Employee management
        - 📈 Attendance statistics and reports  
        - 🔄 Auto-refresh every 30 seconds
        - 📱 Mobile-optimized responsive design
        
        **To access the mobile app:**
        1. Ensure the Mobile API Server is running on port 3000
        2. Open: `http://localhost:3000/mobile` (when implemented)
        3. Or view the demo HTML file directly
        """)
        
        # Show mobile app preview
        st.info("📱 Mobile app HTML demo created at: `mobile_app_demo.html`")
        
    # Integration Instructions
    st.markdown("---")
    st.subheader("🔧 Integration Instructions")
    
    st.markdown("""
    **For Mobile App Developers:**
    
    1. **Start the Mobile API Server:**
       ```bash
       python mobile_api.py
       ```
    
    2. **API Base URL:**
       ```
       http://your-server:3000/api
       ```
    
    3. **Authentication:** Currently no authentication required (add for production)
    
    4. **Response Format:**
       ```json
       {
         "success": true,
         "data": {...},
         "message": "Optional message"
       }
       ```
    
    5. **Error Handling:**
       ```json
       {
         "success": false,
         "error": "Error description"
       }
       ```
    """)
    
    # Live API Testing
    st.markdown("---")
    st.subheader("🧪 Live API Testing")
    
    col_test1, col_test2 = st.columns(2)
    
    with col_test1:
        st.markdown("**Test Employee List API:**")
        if st.button("Get All Employees"):
            try:
                response = requests.get("http://localhost:3000/api/employees")
                data = response.json()
                if data['success']:
                    st.success(f"Found {data['count']} employees")
                    for emp in data['data']:
                        st.write(f"• {emp['name']} ({emp['id']}) - {emp['department']}")
                else:
                    st.error("Failed to load employees")
            except Exception as e:
                st.error(f"API Error: {e}")
    
    with col_test2:
        st.markdown("**Test Today's Stats API:**")
        if st.button("Get Today's Stats"):
            try:
                response = requests.get("http://localhost:3000/api/attendance/today")
                data = response.json()
                if data['success']:
                    stats = data['data']
                    st.metric("Present Today", stats['present_today'])
                    st.metric("Attendance Rate", f"{stats['attendance_rate']}%")
                    
                    if stats['recent_attendance']:
                        st.write("**Recent Check-ins:**")
                        for item in stats['recent_attendance']:
                            st.write(f"• {item['name']} at {item['check_in_time']}")
                else:
                    st.error("Failed to load today's stats")
            except Exception as e:
                st.error(f"API Error: {e}")