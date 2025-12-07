import streamlit as st
import hashlib

# Hardcoded demo credentials
DEMO_USERS = {
    "admin": hashlib.sha256("battery123".encode()).hexdigest(),
    "demo": hashlib.sha256("demo123".encode()).hexdigest()
}

def check_authentication():
    """Check if user is authenticated via session state."""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'username' not in st.session_state:
        st.session_state.username = None
    return st.session_state.authenticated

def login_page():
    """Display login page."""
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <h1>🔋 Battery RUL Prediction System</h1>
        <p style='font-size: 1.2rem; color: #888;'>AI-Powered Lithium-Ion Battery Health Monitoring</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### 🔐 Login")
        username = st.text_input("Username", placeholder="Enter username")
        password = st.text_input("Password", type="password", placeholder="Enter password")
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🚀 Login", use_container_width=True):
                if authenticate(username, password):
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.success(f"Welcome back, {username}! 🎉")
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials. Please try again.")
        
        with col_btn2:
            if st.button("ℹ️ Demo Credentials", use_container_width=True):
                st.info("""
                **Demo Accounts:**
                - Username: `admin` / Password: `battery123`
                - Username: `demo` / Password: `demo123`
                """)

def authenticate(username, password):
    """Authenticate user against demo credentials."""
    if username in DEMO_USERS:
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        return hashed_password == DEMO_USERS[username]
    return False

def logout():
    """Logout current user."""
    st.session_state.authenticated = False
    st.session_state.username = None
    st.rerun()