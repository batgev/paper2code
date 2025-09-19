"""
Repository Finder Agent for Paper2Code

Finds and analyzes relevant GitHub repositories for reference implementations.
"""

from typing import Dict, Any, List, Optional
import asyncio
import aiohttp
import json

from ..config.manager import ConfigManager
from ..utils.logger import get_logger

logger = get_logger(__name__)


class RepositoryFinderAgent:
    """
    AI agent specialized in finding relevant GitHub repositories.
    
    Searches for repositories that might contain reference implementations
    or related code for the research paper being processed.
    """
    
    def __init__(self, config: ConfigManager):
        """
        Initialize repository finder agent.
        
        Args:
            config: Configuration manager instance
        """
        self.config = config
        self.github_api_base = "https://api.github.com"
    
    async def find_repositories(
        self,
        search_terms: List[str],
        analysis_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Find relevant repositories based on search terms and analysis.
        
        Args:
            search_terms: List of search terms extracted from paper
            analysis_results: Results from document analysis
            
        Returns:
            Repository discovery results
        """
        logger.info(f"🔍 Searching for repositories with terms: {search_terms[:5]}...")
        
        try:
            # Extract additional search context
            paper_title = analysis_results.get('document_info', {}).get('title', '')
            algorithms = analysis_results.get('technical_content', {}).get('algorithms', [])
            concepts = analysis_results.get('key_concepts', [])
            
            # Build comprehensive search queries
            search_queries = self._build_search_queries(
                search_terms, paper_title, algorithms, concepts
            )
            
            # Search repositories
            all_repositories = []
            for query in search_queries[:5]:  # Limit to top 5 queries
                repos = await self._search_github_repositories(query)
                all_repositories.extend(repos)
            
            # Filter and rank repositories
            ranked_repositories = self._rank_repositories(
                all_repositories, search_terms, analysis_results
            )
            
            # Get detailed information for top repositories
            detailed_repositories = await self._get_repository_details(
                ranked_repositories[:10]  # Top 10 repositories
            )
            
            result = {
                'search_terms': search_terms,
                'queries_used': search_queries,
                'total_found': len(all_repositories),
                'repositories': detailed_repositories,
                'summary': {
                    'relevant_repos': len([r for r in detailed_repositories if r.get('relevance_score', 0) > 0.5]),
                    'programming_languages': list(set([r.get('language') for r in detailed_repositories if r.get('language')])),
                    'total_stars': sum([r.get('stars', 0) for r in detailed_repositories]),
                }
            }
            
            logger.info(f"✅ Found {len(detailed_repositories)} relevant repositories")
            return result
            
        except Exception as e:
            logger.error(f"❌ Repository discovery failed: {e}")
            # Return empty result instead of failing
            return {
                'search_terms': search_terms,
                'repositories': [],
                'error': str(e),
                'summary': {'relevant_repos': 0, 'programming_languages': [], 'total_stars': 0}
            }
    
    def _build_search_queries(
        self,
        search_terms: List[str],
        paper_title: str,
        algorithms: List[Dict[str, Any]],
        concepts: List[str]
    ) -> List[str]:
        """Build effective GitHub search queries"""
        
        queries = []
        
        # Query from paper title
        if paper_title:
            # Use key words from title
            title_words = [word for word in paper_title.split() if len(word) > 3]
            if title_words:
                queries.append(f"{' '.join(title_words[:3])} implementation")
        
        # Queries from algorithms
        for alg in algorithms[:3]:  # Top 3 algorithms
            alg_name = alg.get('name', '')
            if alg_name:
                clean_name = alg_name.replace('Algorithm', '').strip()
                if clean_name:
                    queries.append(f'"{clean_name}" implementation')
        
        # Queries from key concepts
        for concept in concepts[:3]:  # Top 3 concepts
            if len(concept) > 3:
                queries.append(f'"{concept}" code')
        
        # General queries from search terms
        high_value_terms = [term for term in search_terms if len(term) > 3 and term.lower() not in ['the', 'and', 'for', 'with']]
        for term in high_value_terms[:5]:
            queries.append(f"{term} implementation")
        
        # Combined queries
        if len(high_value_terms) >= 2:
            queries.append(f"{high_value_terms[0]} {high_value_terms[1]}")
        
        return list(set(queries))  # Remove duplicates
    
    async def _search_github_repositories(self, query: str) -> List[Dict[str, Any]]:
        """Search GitHub repositories using API"""
        
        try:
            search_url = f"{self.github_api_base}/search/repositories"
            params = {
                'q': query,
                'sort': 'stars',
                'order': 'desc',
                'per_page': 20  # Limit results per query
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(search_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('items', [])
                    else:
                        logger.warning(f"GitHub API returned status {response.status} for query: {query}")
                        return []
        
        except Exception as e:
            logger.warning(f"Error searching GitHub for query '{query}': {e}")
            return []
    
    def _rank_repositories(
        self,
        repositories: List[Dict[str, Any]],
        search_terms: List[str],
        analysis_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Rank repositories by relevance"""
        
        # Remove duplicates
        unique_repos = {}
        for repo in repositories:
            repo_id = repo.get('id')
            if repo_id and repo_id not in unique_repos:
                unique_repos[repo_id] = repo
        
        repositories = list(unique_repos.values())
        
        # Calculate relevance scores
        for repo in repositories:
            score = self._calculate_relevance_score(repo, search_terms, analysis_results)
            repo['relevance_score'] = score
        
        # Sort by relevance score
        ranked = sorted(repositories, key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        return ranked
    
    def _calculate_relevance_score(
        self,
        repo: Dict[str, Any],
        search_terms: List[str],
        analysis_results: Dict[str, Any]
    ) -> float:
        """Calculate relevance score for a repository"""
        
        score = 0.0
        
        # Basic repository metrics
        stars = repo.get('stargazers_count', 0)
        forks = repo.get('forks_count', 0)
        
        # Popularity score (normalized)
        popularity_score = min((stars + forks) / 1000.0, 1.0)  # Max 1.0
        score += popularity_score * 0.3
        
        # Language relevance
        language_value = repo.get('language')
        language = language_value.lower() if isinstance(language_value, str) else ''
        preferred_languages = ['python', 'pytorch', 'tensorflow']
        if language in preferred_languages:
            score += 0.2
        
        # Name and description relevance
        name = repo.get('name', '').lower()
        description = repo.get('description', '').lower() if repo.get('description') else ''
        
        text_to_check = f"{name} {description}"
        
        # Check for search term matches
        term_matches = 0
        for term in search_terms:
            if term.lower() in text_to_check:
                term_matches += 1
        
        if search_terms:
            term_score = term_matches / len(search_terms)
            score += term_score * 0.3
        
        # Check for algorithm/concept matches
        concepts = analysis_results.get('key_concepts', [])
        concept_matches = 0
        for concept in concepts:
            if concept.lower() in text_to_check:
                concept_matches += 1
        
        if concepts:
            concept_score = concept_matches / len(concepts)
            score += concept_score * 0.2
        
        # Recent activity bonus
        updated_at = repo.get('updated_at', '')
        if updated_at:
            try:
                from datetime import datetime, timezone
                updated_date = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))
                days_ago = (datetime.now(timezone.utc) - updated_date).days
                
                if days_ago < 365:  # Updated within a year
                    freshness_score = max(0, (365 - days_ago) / 365 * 0.1)
                    score += freshness_score
            except:
                pass  # Ignore date parsing errors
        
        return min(score, 1.0)  # Cap at 1.0
    
    async def _get_repository_details(self, repositories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get detailed information for repositories"""
        
        detailed_repos = []
        
        for repo in repositories:
            try:
                # Extract basic info
                detailed_repo = {
                    'name': repo.get('full_name'),
                    'url': repo.get('html_url'),
                    'description': repo.get('description'),
                    'language': repo.get('language'),
                    'stars': repo.get('stargazers_count', 0),
                    'forks': repo.get('forks_count', 0),
                    'updated_at': repo.get('updated_at'),
                    'relevance_score': repo.get('relevance_score', 0),
                    'topics': repo.get('topics', []),
                    'license': repo.get('license', {}).get('name') if repo.get('license') else None,
                }
                
                # Try to get README info (optional)
                try:
                    readme_info = await self._get_readme_info(repo.get('full_name'))
                    detailed_repo['readme_summary'] = readme_info
                except:
                    detailed_repo['readme_summary'] = None
                
                detailed_repos.append(detailed_repo)
                
            except Exception as e:
                logger.warning(f"Error processing repository {repo.get('full_name', 'unknown')}: {e}")
                continue
        
        return detailed_repos
    
    async def _get_readme_info(self, repo_name: str) -> Optional[str]:
        """Get README information for a repository"""
        
        if not repo_name:
            return None
        
        try:
            readme_url = f"{self.github_api_base}/repos/{repo_name}/readme"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(readme_url) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data.get('content', '')
                        
                        # Decode base64 content
                        import base64
                        decoded_content = base64.b64decode(content).decode('utf-8', errors='ignore')
                        
                        # Return first 500 characters as summary
                        return decoded_content[:500] + "..." if len(decoded_content) > 500 else decoded_content
                    else:
                        return None
        
        except Exception as e:
            logger.debug(f"Could not fetch README for {repo_name}: {e}")
            return None
