import math
import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock

from mir.search.communities import search_communities, _build_drill_down_query
from mir.processing.communities import build_tag_communities

@pytest.mark.asyncio
async def test_search_communities_scoring():
    db = AsyncMock()
    
    mock_row = MagicMock()
    mock_row.id = 1
    mock_row.name = "test_community"
    mock_row.type = "tag_cluster"
    mock_row.member_ids = [1, 2, 3]
    mock_row.centroid = [1.0, 0.0]
    
    mock_tag1 = MagicMock()
    mock_tag1.name = "tag1"
    mock_tag2 = MagicMock()
    mock_tag2.name = "tag2"
    
    db.execute.return_value.fetchall.side_effect = [
        [mock_row],
        [mock_tag1, mock_tag2]
    ]
    
    text_processor = MagicMock()
    text_processor.embed.return_value = [np.array([1.0, 0.0], dtype=np.float32)]

    results = await search_communities(db, text_processor, "test query", limit=1)
    
    assert len(results) == 1
    res = results[0]
    assert res.community_id == 1
    assert res.name == "test_community"
    # member_count = 3 -> 1 + 0.1 * log2(4) = 1.2
    assert math.isclose(res.score, 1.2, rel_tol=1e-5)
    assert res.drill_down_query == "/api/v1/search/general?q=tag1+tag2"

def test_drill_down_query():
    top_tags = ["photography", "landscape", "nature"]
    query = _build_drill_down_query(top_tags, "fallback")
    assert query == "/api/v1/search/general?q=photography+landscape+nature"
    
    query_fallback = _build_drill_down_query([], "fallback query")
    assert query_fallback == "/api/v1/search/general?q=fallback+query"

@pytest.mark.asyncio
async def test_clustering_synthetic():
    db = AsyncMock()
    qdrant = AsyncMock()
    qdrant.get_collection_info.return_value = {"vectors_count": 30}
    
    # 3 tight clusters of 10 points each
    np.random.seed(42)
    c1 = np.random.normal(loc=[10, 10], scale=0.1, size=(10, 2))
    c2 = np.random.normal(loc=[-10, -10], scale=0.1, size=(10, 2))
    c3 = np.random.normal(loc=[10, -10], scale=0.1, size=(10, 2))
    vectors = np.vstack([c1, c2, c3]).tolist()
    
    points = []
    for i, v in enumerate(vectors):
        p = MagicMock()
        p.vector = v
        p.id = i
        points.append(p)
        
    qdrant._client.scroll.side_effect = [
        (points, None)
    ]
    
    # Mock the DB queries that happen inside build_tag_communities
    db.execute.return_value.fetchall.return_value = [MagicMock(name=f"tag{i}") for i in range(5)]
    
    count = await build_tag_communities(db, qdrant)
    assert count == 3
