"""
Role-Based Access Control System
Manages user roles, permissions, and navigation based on access levels
"""

import json
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from enum import Enum
import streamlit as st

class UserRole(Enum):
    """User role enumeration"""
    ADMIN = "admin"
    CONTRIBUTOR = "contributor"
    VIEWER = "viewer"
    GUEST = "guest"

class Permission(Enum):
    """Permission enumeration"""
    # Meeting Management
    UPLOAD_MEETINGS = "upload_meetings"
    DELETE_MEETINGS = "delete_meetings"
    EDIT_MEETINGS = "edit_meetings"
    VIEW_MEETINGS = "view_meetings"
    
    # Analysis & Export
    RUN_ANALYSIS = "run_analysis"
    EXPORT_DATA = "export_data"
    GENERATE_REPORTS = "generate_reports"
    
    # User Management
    MANAGE_USERS = "manage_users"
    VIEW_USERS = "view_users"
    
    # System Administration
    SYSTEM_CONFIG = "system_config"
    VIEW_LOGS = "view_logs"
    
    # Advanced Features
    SEMANTIC_SEARCH = "semantic_search"
    EMAIL_SUMMARY = "email_summary"
    AI_ANALYSIS = "ai_analysis"

class RolePermissions:
    """Defines permissions for each role"""
    
    ROLE_PERMISSIONS = {
        UserRole.ADMIN: {
            Permission.UPLOAD_MEETINGS,
            Permission.DELETE_MEETINGS,
            Permission.EDIT_MEETINGS,
            Permission.VIEW_MEETINGS,
            Permission.RUN_ANALYSIS,
            Permission.EXPORT_DATA,
            Permission.GENERATE_REPORTS,
            Permission.MANAGE_USERS,
            Permission.VIEW_USERS,
            Permission.SYSTEM_CONFIG,
            Permission.VIEW_LOGS,
            Permission.SEMANTIC_SEARCH,
            Permission.EMAIL_SUMMARY,
            Permission.AI_ANALYSIS
        },
        UserRole.CONTRIBUTOR: {
            Permission.UPLOAD_MEETINGS,
            Permission.EDIT_MEETINGS,
            Permission.VIEW_MEETINGS,
            Permission.RUN_ANALYSIS,
            Permission.EXPORT_DATA,
            Permission.GENERATE_REPORTS,
            Permission.SEMANTIC_SEARCH,
            Permission.EMAIL_SUMMARY,
            Permission.AI_ANALYSIS
        },
        UserRole.VIEWER: {
            Permission.VIEW_MEETINGS,
            Permission.RUN_ANALYSIS,
            Permission.EXPORT_DATA,
            Permission.SEMANTIC_SEARCH
        },
        UserRole.GUEST: {
            Permission.VIEW_MEETINGS
        }
    }
    
    @classmethod
    def get_permissions(cls, role: UserRole) -> Set[Permission]:
        """Get permissions for a specific role"""
        return cls.ROLE_PERMISSIONS.get(role, set())
    
    @classmethod
    def has_permission(cls, role: UserRole, permission: Permission) -> bool:
        """Check if a role has a specific permission"""
        return permission in cls.get_permissions(role)

class User:
    """User class with role and authentication"""
    
    def __init__(self, username: str, role: UserRole, email: str = None):
        self.username = username
        self.role = role
        self.email = email
        self.created_at = datetime.now()
        self.last_login = None
        self.is_active = True
    
    def to_dict(self) -> Dict:
        """Convert user to dictionary for storage"""
        return {
            'username': self.username,
            'role': self.role.value,
            'email': self.email,
            'created_at': self.created_at.isoformat(),
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'is_active': self.is_active
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'User':
        """Create user from dictionary"""
        user = cls(
            username=data['username'],
            role=UserRole(data['role']),
            email=data.get('email')
        )
        user.created_at = datetime.fromisoformat(data['created_at'])
        if data.get('last_login'):
            user.last_login = datetime.fromisoformat(data['last_login'])
        user.is_active = data.get('is_active', True)
        return user

class UserManager:
    """Manages user authentication and role-based access"""
    
    def __init__(self):
        self.users_file = "users.json"
        self.users: Dict[str, User] = {}
        self.sessions: Dict[str, Dict] = {}
        self._load_users()
        self._create_default_users()
    
    def _load_users(self):
        """Load users from file"""
        try:
            with open(self.users_file, 'r') as f:
                data = json.load(f)
                for user_data in data.values():
                    user = User.from_dict(user_data)
                    self.users[user.username] = user
        except FileNotFoundError:
            pass
    
    def _save_users(self):
        """Save users to file"""
        data = {username: user.to_dict() for username, user in self.users.items()}
        with open(self.users_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _create_default_users(self):
        """Create default users if none exist"""
        if not self.users:
            # Create default admin user
            admin_user = User("admin", UserRole.ADMIN, "admin@verbatimai.com")
            self.users["admin"] = admin_user
            
            # Create Faizan Yousaf as admin
            faizan_user = User("faizan_yousaf", UserRole.ADMIN, "faizanyousaf140@gmail.com")
            self.users["faizan_yousaf"] = faizan_user
            
            # Create Ahsan Saeed as contributor
            ahsan_user = User("ahsan_saeed", UserRole.CONTRIBUTOR, "ahsansaeed1094@gmail.com")
            self.users["ahsan_saeed"] = ahsan_user
            
            # Create some default viewers
            viewer1 = User("viewer1", UserRole.VIEWER, "viewer1@company.com")
            self.users["viewer1"] = viewer1
            
            self._save_users()
    
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user with username and password"""
        # For demo purposes, use simple password hashing
        # In production, use proper password hashing (bcrypt, etc.)
        if username in self.users:
            user = self.users[username]
            if user.is_active:
                # Simple password check (demo only)
                expected_password = self._get_default_password(username)
                if self._hash_password(password) == expected_password:
                    user.last_login = datetime.now()
                    self._save_users()
                    return user
        return None
    
    def _get_default_password(self, username: str) -> str:
        """Get default password for demo users"""
        default_passwords = {
            "admin": self._hash_password("admin123"),
            "faizan_yousaf": self._hash_password("faizan123"),
            "ahsan_saeed": self._hash_password("ahsan123"),
            "viewer1": self._hash_password("viewer123")
        }
        return default_passwords.get(username, self._hash_password("password123"))
    
    def _hash_password(self, password: str) -> str:
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def create_session(self, user: User) -> str:
        """Create a session for authenticated user"""
        session_id = secrets.token_urlsafe(32)
        self.sessions[session_id] = {
            'username': user.username,
            'role': user.role.value,
            'created_at': datetime.now(),
            'expires_at': datetime.now() + timedelta(hours=8)
        }
        return session_id
    
    def get_session_user(self, session_id: str) -> Optional[User]:
        """Get user from session ID"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            if datetime.now() < session['expires_at']:
                username = session['username']
                return self.users.get(username)
            else:
                # Session expired
                del self.sessions[session_id]
        return None
    
    def logout(self, session_id: str):
        """Logout user by removing session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
    
    def has_permission(self, session_id: str, permission: Permission) -> bool:
        """Check if current session has permission"""
        user = self.get_session_user(session_id)
        if user:
            return RolePermissions.has_permission(user.role, permission)
        return False
    
    def get_user_role(self, session_id: str) -> Optional[UserRole]:
        """Get user role from session"""
        user = self.get_session_user(session_id)
        return user.role if user else None

class NavigationManager:
    """Manages navigation based on user roles"""
    
    @staticmethod
    def get_navigation_items(role: UserRole) -> List[Dict]:
        """Get navigation items based on user role"""
        base_items = [
            {"title": "🏠 Dashboard", "icon": "house", "page": "dashboard"},
            {"title": "🎙️ Upload & Transcribe", "icon": "mic", "page": "upload"},
            {"title": "� Real-time Recording", "icon": "record-circle", "page": "recording"},
            {"title": "�📚 Library", "icon": "book", "page": "library"},  # <-- Added Library/History
        ]
        
        if RolePermissions.has_permission(role, Permission.VIEW_MEETINGS):
            base_items.extend([
                {"title": "📊 Analytics", "icon": "graph-up", "page": "analytics"},
                {"title": "🎭 Speaker Sentiment", "icon": "heart", "page": "sentiment"},
            ])
        
        if RolePermissions.has_permission(role, Permission.SEMANTIC_SEARCH):
            base_items.append({"title": "🔍 Semantic Search", "icon": "search", "page": "search"})
        
        if RolePermissions.has_permission(role, Permission.EMAIL_SUMMARY):
            base_items.append({"title": "📧 Email Summary", "icon": "envelope", "page": "email"})
        
        if RolePermissions.has_permission(role, Permission.EXPORT_DATA):
            base_items.append({"title": "📄 Export", "icon": "file-earmark-text", "page": "export"})
        
        if RolePermissions.has_permission(role, Permission.MANAGE_USERS):
            base_items.append({"title": "👥 User Management", "icon": "people", "page": "users"})
        
        if RolePermissions.has_permission(role, Permission.SYSTEM_CONFIG):
            base_items.append({"title": "⚙️ System Settings", "icon": "gear", "page": "settings"})
        
        return base_items
    
    @staticmethod
    def get_page_permissions() -> Dict[str, List[Permission]]:
        """Get required permissions for each page"""
        return {
            "dashboard": [Permission.VIEW_MEETINGS],
            "upload": [Permission.UPLOAD_MEETINGS],
            "recording": [Permission.UPLOAD_MEETINGS],  # Real-time recording uses same permission as upload
            "library": [Permission.VIEW_MEETINGS],
            "analytics": [Permission.VIEW_MEETINGS, Permission.RUN_ANALYSIS],
            "sentiment": [Permission.VIEW_MEETINGS, Permission.RUN_ANALYSIS],
            "search": [Permission.SEMANTIC_SEARCH],
            "email": [Permission.EMAIL_SUMMARY],
            "export": [Permission.EXPORT_DATA],
            "users": [Permission.MANAGE_USERS],
            "settings": [Permission.SYSTEM_CONFIG]
        }

def show_login_page():
    """Display login page"""
    st.title("🔐 VerbatimAI Login")
    
    # Initialize user manager
    if 'user_manager' not in st.session_state:
        st.session_state.user_manager = UserManager()
    
    user_manager = st.session_state.user_manager
    
    # Login form
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button("Login")
        
        if submit_button:
            user = user_manager.authenticate_user(username, password)
            if user:
                session_id = user_manager.create_session(user)
                st.session_state.session_id = session_id
                st.session_state.current_user = user
                st.success(f"✅ Welcome, {user.username}!")
                st.rerun()
            else:
                st.error("❌ Invalid username or password")
    
    # Demo credentials
    st.markdown("### Demo Credentials")
    st.markdown("""
    - **Admin**: username: `admin`, password: `admin123`
    - **Faizan Yousaf**: username: `faizan_yousaf`, password: `faizan123`
    - **Ahsan Saeed**: username: `ahsan_saeed`, password: `ahsan123`
    - **Viewer**: username: `viewer1`, password: `viewer123`
    """)

def check_authentication():
    """Check if user is authenticated"""
    if 'session_id' not in st.session_state:
        return False
    
    user_manager = st.session_state.user_manager
    user = user_manager.get_session_user(st.session_state.session_id)
    
    if not user:
        # Session expired or invalid
        if 'session_id' in st.session_state:
            del st.session_state.session_id
        if 'current_user' in st.session_state:
            del st.session_state.current_user
        return False
    
    return True

def require_permission(permission: Permission):
    """Decorator to require specific permission"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not check_authentication():
                st.error("❌ Please log in to access this feature.")
                return
            
            user_manager = st.session_state.user_manager
            if not user_manager.has_permission(st.session_state.session_id, permission):
                st.error("❌ You don't have permission to access this feature.")
                return
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

def show_user_info():
    """Display current user information"""
    if 'current_user' in st.session_state:
        user = st.session_state.current_user
        st.sidebar.markdown(f"**👤 {user.username}**")
        st.sidebar.markdown(f"**Role**: {user.role.value.title()}")
        st.sidebar.markdown(f"**Email**: {user.email}")
        
        if st.sidebar.button("🚪 Logout"):
            user_manager = st.session_state.user_manager
            user_manager.logout(st.session_state.session_id)
            del st.session_state.session_id
            del st.session_state.current_user
            st.rerun()

def get_current_user_role() -> Optional[UserRole]:
    """Get current user's role"""
    if check_authentication():
        user_manager = st.session_state.user_manager
        return user_manager.get_user_role(st.session_state.session_id)
    return None 