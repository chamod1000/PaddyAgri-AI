"""
Agricultural Knowledge Center Module (core/knowledge_center.py)

Educational and reference module for Sri Lankan Paddy Farming.
Operates completely independent of the Multi-Agent AI reasoning pipeline.
Uses repository abstraction pattern for articles. Zero LLM calls.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


# ══════════════════════════════════════════════
# DATA MODELS
# ══════════════════════════════════════════════
class KnowledgeArticle(BaseModel):
    """Structured agricultural reference article model."""
    article_id: str = Field(..., description="Unique article ID")
    category: str = Field(..., description="Category classification")
    title: str = Field(..., description="Article title")
    summary: str = Field(..., description="Short summary description")
    content: str = Field(..., description="Full reference text content")
    tags: List[str] = Field(default_factory=list, description="Search tags")
    references: List[str] = Field(default_factory=list, description="Citations / DOA publications")
    related_articles: List[str] = Field(default_factory=list, description="IDs of related articles")


class KnowledgeCategory(BaseModel):
    """Knowledge Category descriptor."""
    category_name: str = Field(..., description="Category display name")
    description: str = Field(..., description="Category summary")
    article_count: int = Field(default=0, description="Total articles in category")


class KnowledgeSearchResult(BaseModel):
    """Deterministic search result container."""
    query: str = Field(default="", description="Search query string")
    matched_articles: List[KnowledgeArticle] = Field(default_factory=list, description="Matched articles")
    total_results: int = Field(default=0, description="Total matches found")


# ══════════════════════════════════════════════
# REPOSITORY ABSTRACTION INTERFACE
# ══════════════════════════════════════════════
class IKnowledgeRepository(ABC):
    """Abstract repository interface for Knowledge Center backends (JSON, SQLite, API, CMS)."""

    @abstractmethod
    def list_articles(self) -> List[KnowledgeArticle]:
        pass

    @abstractmethod
    def get_article(self, article_id: str) -> Optional[KnowledgeArticle]:
        pass

    @abstractmethod
    def search_articles(self, query: str = "", category: Optional[str] = None, tag: Optional[str] = None) -> List[KnowledgeArticle]:
        pass


# ══════════════════════════════════════════════
# IN-MEMORY DEFAULT REPOSITORY IMPLEMENTATION
# ══════════════════════════════════════════════
class InMemoryKnowledgeRepository(IKnowledgeRepository):
    """Pre-loaded DOA Sri Lanka paddy farming reference corpus repository."""

    def __init__(self):
        self._articles: Dict[str, KnowledgeArticle] = {}
        self._seed_articles()

    def _seed_articles(self):
        corpus = [
            # 🌾 Paddy Diseases
            KnowledgeArticle(
                article_id="art_blast",
                category="Paddy Diseases",
                title="Paddy Blast (Pyricularia oryzae)",
                summary="Fungal disease causing diamond-shaped leaf lesions and neck rot in high-humidity zones.",
                content=(
                    "Paddy Blast is caused by the fungus Pyricularia oryzae. Symptoms include spindle- or diamond-shaped lesions "
                    "with grey or whitish centers and reddish-brown borders. Severe infections cause collar rot and neck rot, "
                    "preventing grain filling. Management involves planting resistant cultivars (e.g., BG series), avoiding excessive Urea, "
                    "maintaining 3-5 cm standing water depth, and applying DOA-approved fungicides like Tricyclazole or Isoprothiolane."
                ),
                tags=["blast", "fungal", "pyricularia", "lesions", "disease"],
                references=["Sri Lanka Department of Agriculture (DOA) Pathology Circular 2023"],
                related_articles=["art_blight", "art_fungicides"]
            ),
            KnowledgeArticle(
                article_id="art_blight",
                category="Paddy Diseases",
                title="Bacterial Leaf Blight (Xanthomonas oryzae)",
                summary="Bacterial infection causing yellow-to-brown lesions beginning at leaf tips.",
                content=(
                    "Bacterial Leaf Blight (BLB) is caused by Xanthomonas oryzae pv. oryzae. Symptoms appear as water-soaked lesions "
                    "at leaf margins that enlarge into wavy, yellow-to-brown blighted stripes. Bacterial ooze droplets may be visible on young leaves. "
                    "Control involves draining fields periodically, avoiding nitrogen over-fertilization, using Copper-based bactericides, and adopting resistant varieties."
                ),
                tags=["blight", "bacterial", "xanthomonas", "yellowing"],
                references=["DOA Plant Protection Division Technical Bulletin 14"],
                related_articles=["art_blast", "art_ipm"]
            ),
            KnowledgeArticle(
                article_id="art_sheath",
                category="Paddy Diseases",
                title="Sheath Blight (Rhizoctonia solani)",
                summary="Fungal disease affecting lower leaf sheaths near the water line.",
                content=(
                    "Sheath Blight manifests as oval or elliptical greenish-grey spots on leaf sheaths at or near the water line. "
                    "As lesions enlarge, they fuse and turn straw-colored with brown borders. High relative humidity (>85%) and dense planting favor spread. "
                    "Control includes wider seed spacing, balanced NPK split applications, and spraying Hexaconazole or Validamycin."
                ),
                tags=["sheath", "rhizoctonia", "fungal", "humidity"],
                references=["Rice Research and Development Institute (RRDI) Batalagoda Bulletin"],
                related_articles=["art_blast", "art_fungicides"]
            ),

            # 🌱 Fertilizers
            KnowledgeArticle(
                article_id="art_urea",
                category="Fertilizers",
                title="Urea (46% Nitrogen) Application Guide",
                summary="Primary nitrogenous fertilizer for tillering and vegetative canopy development.",
                content=(
                    "Urea provides essential Nitrogen for chlorophyll synthesis and tiller production. Recommended Sri Lankan DOA basal/top-dressing "
                    "rate is 50-60 kg per acre in 3-4 split doses (Basal, 2-3 weeks after sowing, panicle initiation, and heading). "
                    "Do not apply Urea immediately before heavy rains to prevent leaching loss."
                ),
                tags=["urea", "nitrogen", "npk", "dosage", "fertilizer"],
                references=["National Fertilizer Secretariat (NFS) Sri Lanka Guidelines"],
                related_articles=["art_tsp", "art_mop"]
            ),
            KnowledgeArticle(
                article_id="art_tsp",
                category="Fertilizers",
                title="Triple Super Phosphate (46% P2O5) Guide",
                summary="Phosphatic fertilizer for root initiation and early seedling establishment.",
                content=(
                    "Triple Super Phosphate (TSP) supplies Phosphorus crucial for early root vigor and energy transfer (ATP). "
                    "Applied 100% as a basal dose during final land preparation (approx. 20-25 kg/acre). Soil incorporation maximizes uptake."
                ),
                tags=["tsp", "phosphorus", "basal", "fertilizer"],
                references=["DOA Soil Fertility Management Manual"],
                related_articles=["art_urea", "art_mop"]
            ),
            KnowledgeArticle(
                article_id="art_mop",
                category="Fertilizers",
                title="Muriate of Potash (60% K2O) Guide",
                summary="Potassium fertilizer for disease resistance, grain weight, and drought tolerance.",
                content=(
                    "Muriate of Potash (MOP) supplies Potassium which regulates stomatal opening, reduces lodging, and enhances resistance against fungal blast. "
                    "Applied in 2 split doses: basal application and panicle initiation stage (approx. 20-30 kg/acre total)."
                ),
                tags=["mop", "potassium", "potash", "disease resistance"],
                references=["RRDI Batalagoda Agronomy Circular"],
                related_articles=["art_urea", "art_tsp"]
            ),

            # 🛡 Pest & Disease Management
            KnowledgeArticle(
                article_id="art_ipm",
                category="Pest & Disease Management",
                title="Integrated Pest Management (IPM) in Paddy",
                summary="Holistic ecosystem approach combining biological, cultural, and chemical controls.",
                content=(
                    "IPM prioritizes ecological balance: 1) Cultural: synchronized planting, wider spacing; 2) Biological: conserving natural predators "
                    "(spiders, dragonflies, mirid bugs); 3) Chemical: applying targeted pesticides only when economic threshold levels (ETL) are exceeded."
                ),
                tags=["ipm", "pest", "biological", "doa"],
                references=["DOA Plant Protection Division IPM Manual"],
                related_articles=["art_blast", "art_fungicides"]
            ),
            KnowledgeArticle(
                article_id="art_fungicides",
                category="Pest & Disease Management",
                title="DOA Approved Fungicides & Pesticide Act No. 33",
                summary="Regulatory compliance guidelines for safe chemical application in Sri Lanka.",
                content=(
                    "Under Pesticide Act No. 33 of 1980, all agricultural chemicals must be registered with the Registrar of Pesticides (ROP). "
                    "WHO Class Ia/Ib chemicals (e.g. Paraquat, Carbofuran) are strictly banned. Only use registered fungicides with proper PPE."
                ),
                tags=["fungicides", "pesticide act", "regulatory", "safety"],
                references=["Registrar of Pesticides (ROP) Approved Chemical List 2024"],
                related_articles=["art_ipm", "art_blast"]
            ),

            # 🌦 Cultivation Best Practices
            KnowledgeArticle(
                article_id="art_water_mgmt",
                category="Cultivation Best Practices",
                title="Paddy Field Water Management & Alternate Wetting and Drying (AWD)",
                summary="Efficient irrigation techniques for water conservation and root aeration.",
                content=(
                    "Maintain 2-5 cm standing water during seedling establishment and tillering. Adopt Alternate Wetting and Drying (AWD) "
                    "after tillering to promote deep root growth, save 30% irrigation water, and suppress methane emissions."
                ),
                tags=["water", "irrigation", "awd", "best practices"],
                references=["Department of Irrigation Sri Lanka Guidelines"],
                related_articles=["art_ipm", "art_urea"]
            )
        ]
        for a in corpus:
            self._articles[a.article_id] = a

    def list_articles(self) -> List[KnowledgeArticle]:
        return list(self._articles.values())

    def get_article(self, article_id: str) -> Optional[KnowledgeArticle]:
        return self._articles.get(article_id)

    def search_articles(self, query: str = "", category: Optional[str] = None, tag: Optional[str] = None) -> List[KnowledgeArticle]:
        results = list(self._articles.values())

        if category and category.strip():
            results = [a for a in results if a.category.lower() == category.lower().strip()]

        if tag and tag.strip():
            results = [a for a in results if tag.lower().strip() in [t.lower() for t in a.tags]]

        if query and query.strip():
            q = query.lower().strip()
            matched = []
            for a in results:
                if (q in a.title.lower() or
                    q in a.summary.lower() or
                    q in a.content.lower() or
                    any(q in t.lower() for t in a.tags)):
                    matched.append(a)
            results = matched

        return results


# ══════════════════════════════════════════════
# KNOWLEDGE CENTER SERVICE
# ══════════════════════════════════════════════
class KnowledgeCenter:
    """
    High-level Educational & Reference Knowledge Center Service.
    Operates independently of AI reasoning pipeline.
    """

    def __init__(self, repository: Optional[IKnowledgeRepository] = None):
        self.repository = repository if repository else InMemoryKnowledgeRepository()

    def search(self, query: str = "", category: Optional[str] = None) -> KnowledgeSearchResult:
        articles = self.repository.search_articles(query=query, category=category)
        return KnowledgeSearchResult(
            query=query,
            matched_articles=articles,
            total_results=len(articles)
        )

    def get_categories(self) -> List[KnowledgeCategory]:
        articles = self.repository.list_articles()
        cat_counts: Dict[str, int] = {}
        for a in articles:
            cat_counts[a.category] = cat_counts.get(a.category, 0) + 1

        cats = []
        for cat_name, count in cat_counts.items():
            cats.append(KnowledgeCategory(
                category_name=cat_name,
                description=f"DOA Guidelines and reference articles for {cat_name}.",
                article_count=count
            ))
        return cats
