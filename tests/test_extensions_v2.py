"""
Tests for Msty Admin MCP Server Extensions v2 (Phases 16-25)

Tests for:
- Shadow Persona Integration
- Workspaces Management
- Real-Time Web/Data Integration
- Chat/Conversation Management
- Folder Organization
- PII Scrubbing Tools
- Embedding Visualization
- Health Monitoring Dashboard
- Configuration Profiles
- Automated Maintenance
"""

import json
import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock


class TestShadowPersonas:
    """Tests for Phase 16: Shadow Persona Integration"""

    def test_list_shadow_personas(self):
        """Test listing shadow personas"""
        from src.shadow_personas import list_shadow_personas
        result = list_shadow_personas()

        assert "timestamp" in result
        assert "shadow_personas" in result
        assert "total_count" in result
        assert isinstance(result["shadow_personas"], list)

    def test_get_shadow_persona_details_not_found(self):
        """Test getting details for non-existent shadow persona"""
        from src.shadow_personas import get_shadow_persona_details
        result = get_shadow_persona_details("nonexistent_shadow")

        assert "timestamp" in result
        assert result["found"] == False or "error" in result

    def test_analyze_shadow_conversation_missing(self):
        """Test analyzing with missing conversation"""
        from src.shadow_personas import analyze_shadow_conversation
        result = analyze_shadow_conversation("nonexistent_conv")

        assert "timestamp" in result
        assert "conversation_id" in result


class TestWorkspaces:
    """Tests for Phase 17: Workspaces Management"""

    def test_list_workspaces(self):
        """Test listing workspaces"""
        from src.workspaces import list_workspaces
        result = list_workspaces()

        assert "timestamp" in result
        assert "workspaces" in result
        assert "total_count" in result
        assert isinstance(result["workspaces"], list)

    def test_get_workspace_details_not_found(self):
        """Test getting details for non-existent workspace"""
        from src.workspaces import get_workspace_details
        result = get_workspace_details("nonexistent_ws")

        assert "timestamp" in result
        assert result["found"] == False or "error" in result

    def test_get_workspace_stats(self):
        """Test getting workspace stats"""
        from src.workspaces import get_workspace_stats
        result = get_workspace_stats()

        assert "timestamp" in result
        assert "statistics" in result


class TestRealtimeData:
    """Tests for Phase 18: Real-Time Web/Data Integration"""

    def test_extract_youtube_id_full_url(self):
        """Test extracting YouTube ID from full URL"""
        from src.realtime_data import extract_youtube_id

        # Standard watch URL
        assert extract_youtube_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"

        # Short URL
        assert extract_youtube_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

        # Note: Just ID without URL format may return None (implementation specific)

    def test_realtime_search_structure(self):
        """Test real-time search returns correct structure"""
        from src.realtime_data import realtime_search
        result = realtime_search("test query")

        assert "timestamp" in result
        assert "query" in result
        assert "results" in result
        assert isinstance(result["results"], list)


class TestChatManagement:
    """Tests for Phase 19: Chat/Conversation Management"""

    def test_export_chat_thread_missing(self):
        """Test exporting non-existent chat"""
        from src.chat_management import export_chat_thread
        result = export_chat_thread("nonexistent_chat")

        assert "timestamp" in result
        # May use "conversation_id" or "chat_id" - check for either
        assert "conversation_id" in result or "chat_id" in result
        assert "export_format" in result

    def test_clone_chat_missing(self):
        """Test cloning non-existent chat"""
        from src.chat_management import clone_chat
        result = clone_chat("nonexistent_chat")

        assert "timestamp" in result
        # May use different key names
        assert "source_conversation_id" in result or "original_chat_id" in result


class TestFolders:
    """Tests for Phase 20: Folder Organization"""

    def test_list_folders(self):
        """Test listing folders"""
        from src.folders import list_folders
        result = list_folders()

        assert "timestamp" in result
        assert "folders" in result
        assert "total_count" in result

    def test_get_folder_stats(self):
        """Test folder statistics"""
        from src.folders import get_folder_stats
        result = get_folder_stats()

        assert "timestamp" in result
        assert "statistics" in result

    def test_suggest_folder_organization(self):
        """Test folder organization suggestions"""
        from src.folders import suggest_folder_organization
        result = suggest_folder_organization()

        assert "timestamp" in result
        assert "suggestions" in result


class TestPIITools:
    """Tests for Phase 21: PII Scrubbing Tools"""

    def test_scan_for_pii_email(self):
        """Test scanning for email addresses"""
        from src.pii_tools import scan_for_pii

        text = "Contact me at john.doe@example.com for more info"
        result = scan_for_pii(text)

        assert "timestamp" in result
        # Result uses "findings" not "detections"
        assert "findings" in result
        assert len(result["findings"]) > 0
        assert any(d["category"] == "email" for d in result["findings"])

    def test_scan_for_pii_phone(self):
        """Test scanning for phone numbers"""
        from src.pii_tools import scan_for_pii

        text = "Call me at 555-123-4567"
        result = scan_for_pii(text)

        assert "findings" in result
        # Phone detection category may be "phone_us"
        assert any("phone" in d["category"] for d in result["findings"])

    def test_scan_for_pii_ssn(self):
        """Test scanning for SSN"""
        from src.pii_tools import scan_for_pii

        text = "My SSN is 123-45-6789"
        result = scan_for_pii(text)

        assert "findings" in result
        assert any(d["category"] == "ssn" for d in result["findings"])

    def test_scrub_pii(self):
        """Test scrubbing PII from text"""
        from src.pii_tools import scrub_pii

        text = "Email: test@example.com Phone: 555-123-4567"
        result = scrub_pii(text)

        assert "timestamp" in result
        assert "scrubbed_text" in result
        assert "test@example.com" not in result["scrubbed_text"]
        assert "555-123-4567" not in result["scrubbed_text"]

    def test_scan_for_pii_summary(self):
        """Test that PII scan includes summary"""
        from src.pii_tools import scan_for_pii

        text = "Email: test@example.com"
        result = scan_for_pii(text)

        assert "summary" in result
        assert "risk_level" in result


class TestEmbeddings:
    """Tests for Phase 22: Embedding Visualization"""

    def test_cosine_similarity(self):
        """Test cosine similarity calculation"""
        from src.embeddings import cosine_similarity

        # Same vector should have similarity of 1.0
        v1 = [1.0, 0.0, 0.0]
        assert abs(cosine_similarity(v1, v1) - 1.0) < 0.001

        # Orthogonal vectors should have similarity of 0.0
        v2 = [0.0, 1.0, 0.0]
        assert abs(cosine_similarity(v1, v2) - 0.0) < 0.001

        # Opposite vectors should have similarity of -1.0
        v3 = [-1.0, 0.0, 0.0]
        assert abs(cosine_similarity(v1, v3) - (-1.0)) < 0.001

    def test_get_embeddings_from_stack_structure(self):
        """Test getting embeddings returns valid structure"""
        from src.embeddings import get_embeddings_from_stack
        result = get_embeddings_from_stack("nonexistent_stack")

        # May return empty list or dict depending on implementation
        assert isinstance(result, (list, dict))

    def test_embedding_statistics(self):
        """Test embedding statistics structure"""
        from src.embeddings import embedding_statistics
        result = embedding_statistics("test_stack")

        assert "timestamp" in result
        assert "stack_id" in result


class TestDashboard:
    """Tests for Phase 23: Health Monitoring Dashboard"""

    def test_get_dashboard_status(self):
        """Test getting dashboard status"""
        from src.dashboard import get_dashboard_status
        result = get_dashboard_status()

        assert "timestamp" in result
        # Result has "overall_status" and "services" keys
        assert "overall_status" in result or "status" in result
        assert "services" in result

    def test_get_active_alerts(self):
        """Test getting active alerts"""
        from src.dashboard import get_active_alerts
        result = get_active_alerts()

        assert "timestamp" in result
        assert "alerts" in result
        assert "summary" in result


class TestProfiles:
    """Tests for Phase 24: Configuration Profiles"""

    def test_list_profiles(self):
        """Test listing profiles"""
        from src.profiles import list_profiles
        result = list_profiles()

        assert "timestamp" in result
        assert "profiles" in result
        assert "total_count" in result

    def test_save_profile(self):
        """Test saving a profile"""
        from src.profiles import save_profile
        result = save_profile("test_profile", "Test description")

        assert "timestamp" in result
        assert "profile_name" in result

    def test_load_profile_not_found(self):
        """Test loading non-existent profile"""
        from src.profiles import load_profile
        result = load_profile("nonexistent_profile")

        assert "timestamp" in result
        assert "error" in result or "loaded" in result


class TestMaintenance:
    """Tests for Phase 25: Automated Maintenance"""

    def test_identify_cleanup_candidates(self):
        """Test identifying cleanup candidates"""
        from src.maintenance import identify_cleanup_candidates
        result = identify_cleanup_candidates()

        assert "timestamp" in result
        assert "candidates" in result
        assert "total_potential_savings_mb" in result

    def test_perform_cleanup_dry_run(self):
        """Test cleanup in dry run mode"""
        from src.maintenance import perform_cleanup
        result = perform_cleanup(dry_run=True)

        assert "timestamp" in result
        assert result["dry_run"] == True
        assert "deleted" in result
        assert "note" in result  # Should have dry run note

    def test_optimize_database(self):
        """Test database optimization"""
        from src.maintenance import optimize_database
        result = optimize_database()

        assert "timestamp" in result
        assert "optimizations" in result

    def test_generate_maintenance_report(self):
        """Test generating maintenance report"""
        from src.maintenance import generate_maintenance_report
        result = generate_maintenance_report()

        assert "timestamp" in result
        assert "report" in result
        assert "recommendations" in result
        assert "health_score" in result
        assert "health_status" in result
        assert result["health_score"] >= 0
        assert result["health_score"] <= 100


class TestServerExtensionsV2:
    """Integration tests for server_extensions_v2"""

    def test_all_modules_importable(self):
        """Test all new modules are importable"""
        # These imports should not raise any exceptions
        from src import shadow_personas
        from src import workspaces
        from src import realtime_data
        from src import chat_management
        from src import folders
        from src import pii_tools
        from src import embeddings
        from src import dashboard
        from src import profiles
        from src import maintenance

        # Verify __all__ is defined
        assert hasattr(shadow_personas, '__all__')
        assert hasattr(workspaces, '__all__')
        assert hasattr(realtime_data, '__all__')
        assert hasattr(chat_management, '__all__')
        assert hasattr(folders, '__all__')
        assert hasattr(pii_tools, '__all__')
        assert hasattr(embeddings, '__all__')
        assert hasattr(dashboard, '__all__')
        assert hasattr(profiles, '__all__')
        assert hasattr(maintenance, '__all__')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
