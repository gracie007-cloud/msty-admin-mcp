"""
Msty Admin MCP Server Extensions v8.0.0

New tools for Phase 16+:
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

These extensions add 36 new tools to the server (Phases 16-25).
"""

import json
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

from mcp.server.fastmcp import FastMCP

# Import new modules (Phase 16-25)
from .shadow_personas import (
    list_shadow_personas,
    get_shadow_persona_details,
    analyze_shadow_conversation,
    synthesize_shadow_responses,
    compare_shadow_responses,
)
from .workspaces import (
    list_workspaces,
    get_workspace_details,
    get_workspace_stats,
    export_workspace,
)
from .realtime_data import (
    realtime_search,
    realtime_fetch,
    realtime_youtube_transcript,
)
from .chat_management import (
    export_chat_thread,
    clone_chat,
    branch_chat,
    merge_chats,
)
from .folders import (
    list_folders,
    get_folder_details,
    get_folder_stats,
    suggest_folder_organization,
)
from .pii_tools import (
    scan_for_pii,
    scrub_pii,
    generate_pii_report,
)
from .embeddings import (
    get_embeddings_from_stack,
    visualize_embeddings,
    cluster_embeddings,
    compare_embeddings,
)
from .dashboard import (
    check_service_health,
    get_dashboard_status,
    get_active_alerts,
)
from .profiles import (
    list_profiles,
    save_profile,
    load_profile,
    compare_profiles,
)
from .maintenance import (
    identify_cleanup_candidates,
    perform_cleanup,
    optimize_database,
)

logger = logging.getLogger("msty-admin-mcp")


def register_extension_tools_v2(mcp: FastMCP):
    """Register all v2 extension tools with the MCP server (Phases 16-25)."""

    # =========================================================================
    # Phase 16: Shadow Persona Integration
    # =========================================================================

    @mcp.tool()
    def shadow_list() -> str:
        """
        List all configured shadow personas.

        Returns:
            JSON with shadow personas and their configurations
        """
        result = list_shadow_personas()
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def shadow_details(persona_id: str) -> str:
        """
        Get detailed information about a specific shadow persona.

        Args:
            persona_id: The shadow persona identifier or name

        Returns:
            JSON with shadow persona configuration and capabilities
        """
        result = get_shadow_persona_details(persona_id)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def shadow_analyze(
        conversation_id: str,
        shadow_persona_id: Optional[str] = None
    ) -> str:
        """
        Analyze a conversation from a shadow persona's perspective.

        Args:
            conversation_id: The conversation to analyze
            shadow_persona_id: Specific shadow to use (optional)

        Returns:
            JSON with shadow analysis results
        """
        result = analyze_shadow_conversation(conversation_id, shadow_persona_id)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def shadow_synthesize(
        conversation_id: str,
        include_main: bool = True
    ) -> str:
        """
        Synthesize insights from shadow persona observations.

        Args:
            conversation_id: The conversation to synthesize
            include_main: Include main conversation responses

        Returns:
            JSON with synthesized shadow insights
        """
        result = synthesize_shadow_responses(conversation_id, include_main)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def shadow_compare(
        conversation_id: str,
        shadow_ids: Optional[List[str]] = None
    ) -> str:
        """
        Compare responses between main conversation and shadow personas.

        Args:
            conversation_id: The conversation to compare
            shadow_ids: Specific shadows to compare (optional)

        Returns:
            JSON with comparison results
        """
        result = compare_shadow_responses(conversation_id, shadow_ids)
        return json.dumps(result, indent=2, default=str)

    # =========================================================================
    # Phase 17: Workspaces Management
    # =========================================================================

    @mcp.tool()
    def workspace_list() -> str:
        """
        List all available workspaces.

        Returns:
            JSON with workspaces list and metadata
        """
        result = list_workspaces()
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def workspace_details(workspace_id: str) -> str:
        """
        Get detailed information about a specific workspace.

        Args:
            workspace_id: The workspace identifier

        Returns:
            JSON with workspace details
        """
        result = get_workspace_details(workspace_id)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def workspace_stats(workspace_id: Optional[str] = None) -> str:
        """
        Get statistics for a workspace or all workspaces.

        Args:
            workspace_id: Specific workspace (optional, all if None)

        Returns:
            JSON with workspace statistics
        """
        result = get_workspace_stats(workspace_id)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def workspace_export(
        workspace_id: str,
        export_format: str = "json",
        include_conversations: bool = True,
        include_personas: bool = True,
        include_knowledge_stacks: bool = True
    ) -> str:
        """
        Export a workspace for backup or migration.

        Args:
            workspace_id: The workspace to export
            export_format: Output format (json, zip)
            include_conversations: Include conversation history
            include_personas: Include persona configurations
            include_knowledge_stacks: Include knowledge stack data

        Returns:
            JSON with export data or file path
        """
        result = export_workspace(
            workspace_id, export_format,
            include_conversations, include_personas, include_knowledge_stacks
        )
        return json.dumps(result, indent=2, default=str)

    # =========================================================================
    # Phase 18: Real-Time Web/Data Integration
    # =========================================================================

    @mcp.tool()
    def rt_search(query: str, max_results: int = 5) -> str:
        """
        Perform real-time web search.

        Args:
            query: Search query
            max_results: Maximum results to return

        Returns:
            JSON with search results
        """
        result = realtime_search(query, max_results)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def rt_fetch(url: str, extract_text: bool = True) -> str:
        """
        Fetch and extract content from a URL.

        Args:
            url: URL to fetch
            extract_text: Extract main text content

        Returns:
            JSON with fetched content
        """
        result = realtime_fetch(url, extract_text)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def rt_youtube(video_url: str, include_metadata: bool = True) -> str:
        """
        Extract transcript from a YouTube video.

        Args:
            video_url: YouTube video URL or ID
            include_metadata: Include video metadata

        Returns:
            JSON with transcript and optional metadata
        """
        result = realtime_youtube_transcript(video_url, include_metadata)
        return json.dumps(result, indent=2, default=str)

    # =========================================================================
    # Phase 19: Chat/Conversation Management
    # =========================================================================

    @mcp.tool()
    def chat_export(
        chat_id: str,
        export_format: str = "markdown",
        include_metadata: bool = True
    ) -> str:
        """
        Export a chat thread to various formats.

        Args:
            chat_id: The chat to export
            export_format: Output format (markdown, json, text, html)
            include_metadata: Include chat metadata

        Returns:
            JSON with exported content or file path
        """
        result = export_chat_thread(chat_id, export_format, include_metadata)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def chat_clone(chat_id: str, new_title: Optional[str] = None) -> str:
        """
        Clone a chat conversation.

        Args:
            chat_id: The chat to clone
            new_title: Title for the cloned chat

        Returns:
            JSON with clone result
        """
        result = clone_chat(chat_id, new_title)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def chat_branch(
        chat_id: str,
        from_message_index: int,
        branch_title: Optional[str] = None
    ) -> str:
        """
        Create a branch from a specific point in conversation.

        Args:
            chat_id: The chat to branch from
            from_message_index: Message index to branch from
            branch_title: Title for the branch

        Returns:
            JSON with branch result
        """
        result = branch_chat(chat_id, from_message_index, branch_title)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def chat_merge(
        primary_chat_id: str,
        secondary_chat_id: str,
        merge_strategy: str = "append"
    ) -> str:
        """
        Merge two chat conversations.

        Args:
            primary_chat_id: Primary chat (base)
            secondary_chat_id: Chat to merge in
            merge_strategy: How to merge (append, interleave)

        Returns:
            JSON with merge result
        """
        result = merge_chats(primary_chat_id, secondary_chat_id, merge_strategy)
        return json.dumps(result, indent=2, default=str)

    # =========================================================================
    # Phase 20: Folder Organization
    # =========================================================================

    @mcp.tool()
    def folder_list() -> str:
        """
        List all conversation folders.

        Returns:
            JSON with folders and their contents
        """
        result = list_folders()
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def folder_details(folder_id: str) -> str:
        """
        Get detailed information about a folder.

        Args:
            folder_id: The folder identifier

        Returns:
            JSON with folder details and contents
        """
        result = get_folder_details(folder_id)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def folder_stats() -> str:
        """
        Get statistics about folder organization.

        Returns:
            JSON with folder statistics
        """
        result = get_folder_stats()
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def folder_suggest() -> str:
        """
        Get suggestions for organizing conversations into folders.

        Returns:
            JSON with organization suggestions
        """
        result = suggest_folder_organization()
        return json.dumps(result, indent=2, default=str)

    # =========================================================================
    # Phase 21: PII Scrubbing Tools
    # =========================================================================

    @mcp.tool()
    def pii_scan(
        text: str,
        pii_types: Optional[List[str]] = None
    ) -> str:
        """
        Scan text for Personally Identifiable Information.

        Args:
            text: Text to scan
            pii_types: Specific PII types to scan for (optional)

        Returns:
            JSON with detected PII
        """
        result = scan_for_pii(text, pii_types)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def pii_scrub(
        text: str,
        replacement: str = "[REDACTED]",
        pii_types: Optional[List[str]] = None
    ) -> str:
        """
        Remove or mask PII from text.

        Args:
            text: Text to scrub
            replacement: Replacement text for PII
            pii_types: Specific PII types to scrub (optional)

        Returns:
            JSON with scrubbed text
        """
        result = scrub_pii(text, replacement, pii_types)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def pii_report(conversation_id: str) -> str:
        """
        Generate a PII report for a conversation.

        Args:
            conversation_id: The conversation to analyze

        Returns:
            JSON with PII analysis report
        """
        result = generate_pii_report(conversation_id)
        return json.dumps(result, indent=2, default=str)

    # =========================================================================
    # Phase 22: Embedding Visualization
    # =========================================================================

    @mcp.tool()
    def embedding_get(stack_id: str, limit: int = 100) -> str:
        """
        Get embeddings from a Knowledge Stack.

        Args:
            stack_id: The stack to get embeddings from
            limit: Maximum number of embeddings

        Returns:
            JSON with embeddings data
        """
        result = get_embeddings_from_stack(stack_id, limit)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def embedding_visualize(stack_id: str, method: str = "pca") -> str:
        """
        Generate visualization data for embeddings.

        Args:
            stack_id: The stack to visualize
            method: Visualization method (pca, tsne, umap)

        Returns:
            JSON with visualization coordinates
        """
        result = visualize_embeddings(stack_id, method)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def embedding_cluster(stack_id: str, num_clusters: int = 5) -> str:
        """
        Cluster embeddings for topic discovery.

        Args:
            stack_id: The stack to cluster
            num_clusters: Number of clusters

        Returns:
            JSON with cluster assignments
        """
        result = cluster_embeddings(stack_id, num_clusters)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def embedding_compare(stack_id: str, doc_id_1: str, doc_id_2: str) -> str:
        """
        Compare embeddings of two documents.

        Args:
            stack_id: The stack containing the documents
            doc_id_1: First document ID
            doc_id_2: Second document ID

        Returns:
            JSON with similarity analysis
        """
        result = compare_embeddings(stack_id, doc_id_1, doc_id_2)
        return json.dumps(result, indent=2, default=str)

    # =========================================================================
    # Phase 23: Health Monitoring Dashboard
    # =========================================================================

    @mcp.tool()
    def health_check(service_name: Optional[str] = None) -> str:
        """
        Check health of Msty services.

        Args:
            service_name: Specific service to check (optional)

        Returns:
            JSON with service health status
        """
        result = check_service_health(service_name)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def health_dashboard() -> str:
        """
        Get comprehensive dashboard status.

        Returns:
            JSON with full system status overview
        """
        result = get_dashboard_status()
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def health_alerts() -> str:
        """
        Get active health alerts and warnings.

        Returns:
            JSON with active alerts
        """
        result = get_active_alerts()
        return json.dumps(result, indent=2, default=str)

    # =========================================================================
    # Phase 24: Configuration Profiles
    # =========================================================================

    @mcp.tool()
    def profile_list() -> str:
        """
        List all saved configuration profiles.

        Returns:
            JSON with profiles list
        """
        result = list_profiles()
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def profile_save(
        name: str,
        description: str = "",
        include_personas: bool = True,
        include_tools: bool = True,
        include_settings: bool = True,
        include_prompts: bool = True
    ) -> str:
        """
        Save current configuration as a profile.

        Args:
            name: Profile name
            description: Profile description
            include_personas: Include persona configs
            include_tools: Include MCP tool configs
            include_settings: Include app settings
            include_prompts: Include saved prompts

        Returns:
            JSON with save result
        """
        result = save_profile(
            name, description,
            include_personas, include_tools,
            include_settings, include_prompts
        )
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def profile_load(profile_id: str, dry_run: bool = True) -> str:
        """
        Load a saved configuration profile.

        Args:
            profile_id: Profile to load
            dry_run: Preview only (default: True)

        Returns:
            JSON with profile data or load result
        """
        result = load_profile(profile_id, dry_run)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def profile_compare(profile_id_1: str, profile_id_2: str) -> str:
        """
        Compare two configuration profiles.

        Args:
            profile_id_1: First profile
            profile_id_2: Second profile

        Returns:
            JSON with comparison results
        """
        result = compare_profiles(profile_id_1, profile_id_2)
        return json.dumps(result, indent=2, default=str)

    # =========================================================================
    # Phase 25: Automated Maintenance
    # =========================================================================

    @mcp.tool()
    def maintenance_identify() -> str:
        """
        Identify files and data that can be cleaned up.

        Returns:
            JSON with cleanup candidates and potential savings
        """
        result = identify_cleanup_candidates()
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def maintenance_cleanup(
        cleanup_types: Optional[List[str]] = None,
        max_age_days: int = 30,
        dry_run: bool = True
    ) -> str:
        """
        Perform cleanup operations.

        Args:
            cleanup_types: Types to clean (cache, log, old_export, incomplete_download)
            max_age_days: Maximum age for time-based cleanup
            dry_run: Preview only (default: True)

        Returns:
            JSON with cleanup results
        """
        result = perform_cleanup(cleanup_types, max_age_days, dry_run)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def maintenance_optimize() -> str:
        """
        Optimize the Msty database.

        Returns:
            JSON with optimization results
        """
        result = optimize_database()
        return json.dumps(result, indent=2, default=str)

    logger.info("Registered 36 extension tools (Phases 16-25)")


__all__ = ["register_extension_tools_v2"]
