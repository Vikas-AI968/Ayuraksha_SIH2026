"""
Synthetic Data Generator for IP-SAKTI Sahayak Phase-1.
Generates 32 synthetic authoritative-style documents in JSON/TXT format.
All documents are strictly marked with source_type = 'synthetic' and URI synthetic://
"""
import os
import json

SYNTHETIC_DIR = "data/synthetic"
os.makedirs(SYNTHETIC_DIR, exist_ok=True)

DOCUMENTS = [
    # 1. Patents - Section 3(p) & Traditional Knowledge
    {
        "document_id": "SYN-PAT-001",
        "title": "Guidelines for Patent Applications in Traditional Knowledge & Ayurveda",
        "authority": "IP-SAKTI Synthetic Patent Office",
        "jurisdiction": "India",
        "domain": "Patents",
        "document_type": "Guidance",
        "effective_date": "2025-01-15",
        "source_url": "synthetic://patent-office/guidelines-tk-ayurveda-2025",
        "source_type": "synthetic",
        "version": "1.0",
        "content": """CHAPTER I: INTELLECTUAL PROPERTY AND TRADITIONAL KNOWLEDGE IN INDIA

Section 1. Purpose and Scope
These synthetic guidelines outline the patentability criteria for inventions relating to Ayurvedic formulations, traditional herbal knowledge, and medicinal plant extracts under the Indian Patents Act, 1970.

Section 2. Section 3(p) Exclusion
Under Section 3(p) of the Patents Act, 1970, an invention which in effect is traditional knowledge or which is an aggregation or duplication of known properties of traditionally known component or components is not an patentable invention. 

Section 3. Patentability of Synergistic Combinations
An Ayurvedic formulation combining traditional herbs (such as Ashwagandha and Turmeric) may be patentable if:
(a) The applicant demonstrates a non-obvious synergistic therapeutic effect exceeding the additive sum of individual components.
(b) Experimental comparative data shows unexpectedly superior bio-availability or reduced toxicity.
(c) The exact active extract fraction is novel, structurally characterized, and isolated through a non-conventional process.

Section 4. TKDL Prior Art Citation
Examiners shall search the Traditional Knowledge Digital Library (TKDL) database. If a claimed formulation or therapeutic usage is documented in classical texts (e.g. Charaka Samhita, Sushruta Samhita, Astanga Hridaya) indexed in TKDL, the claim shall be rejected under Section 3(p) and Section 3(d) for lacking novelty and inventive step."""
    },
    
    # 2. Patents - Phytopharmaceuticals & Novel Formulations
    {
        "document_id": "SYN-PAT-002",
        "title": "Patentability Guidance for Novel Phytopharmaceutical Formulations",
        "authority": "IP-SAKTI Synthetic Patent Office",
        "jurisdiction": "India",
        "domain": "Patents",
        "document_type": "Guidance",
        "effective_date": "2026-02-01",
        "source_url": "synthetic://patent-office/phytopharmaceutical-guidance-2026",
        "source_type": "synthetic",
        "version": "2.0",
        "content": """CHAPTER II: PHYTOPHARMACEUTICAL PATENT REQUIREMENTS

Section 1. Definition of Phytopharmaceutical Drug
A phytopharmaceutical drug is defined as a purified and standardized fraction with defined active chemical markers derived from a plant or part of a plant, intended for internal or external use of human beings or animals.

Section 2. Section 3(d) and Enhanced Efficacy
Where a phytopharmaceutical is derived from a known Ayurvedic plant material, Section 3(d) requires clear evidence of significantly enhanced therapeutic efficacy. Mere isolation of an active ingredient without comparative clinical or bio-assay data demonstrating superior efficacy over the raw herbal substance is insufficient to overcome Section 3(d).

Section 3. Prior Informed Consent and NBA Clearance
Applications claiming phytopharmaceutical products or processes using Indian biological resources must append a mandatory clearance certificate from the National Biodiversity Authority (NBA) prior to the grant of the patent as mandated by Section 6 of the Biological Diversity Act, 2002."""
    },

    # 3. Geographical Indications - Ayurveda GI Guidance
    {
        "document_id": "SYN-GI-001",
        "title": "Geographical Indications Registration for Regional Ayurvedic Formulations",
        "authority": "IP-SAKTI Synthetic GI Registry",
        "jurisdiction": "India",
        "domain": "Geographical Indications",
        "document_type": "Rule",
        "effective_date": "2024-06-10",
        "source_url": "synthetic://gi-registry/ayurveda-gi-guidance-2024",
        "source_type": "synthetic",
        "version": "1.0",
        "content": """REGULATION OF GEOGRAPHICAL INDICATIONS FOR AYURVEDIC PRODUCTS

Section 1. Eligibility Criteria for Ayurvedic GI Registration
A regional traditional Ayurvedic preparation (e.g. Navara Rice, Kerala Ayurveda Oil, Malabar Herbs) is eligible for Geographical Indication registration under the Geographical Indications of Goods (Registration and Protection) Act, 1999 if:
(a) The product possesses specific qualities, reputation, or characteristics attributable to its geographical origin in India.
(b) The preparation relies on specific local ecological conditions, traditional processing techniques, or indigenous medicinal plant strains native to the demarcated region.

Section 2. Protection Against Generic Misuse
Registration of an Ayurvedic GI grants exclusive rights to authorized producers registered within the specified geographic area. Unauthorized use of the GI name on generic or synthetic substitutes constitutes infringement under Section 22 of the GI Act."""
    },

    # 4. Geographical Indications - Regional Herbal Product Scenario
    {
        "document_id": "SYN-GI-002",
        "title": "Case Analysis: Regional Herbal GI Protection and Community Rights",
        "authority": "IP-SAKTI Synthetic GI Registry",
        "jurisdiction": "India",
        "domain": "Geographical Indications",
        "document_type": "Scenario",
        "effective_date": "2025-09-20",
        "source_url": "synthetic://gi-registry/regional-herbal-gi-scenario-2025",
        "source_type": "synthetic",
        "version": "1.0",
        "content": """SCENARIO ANALYSIS: KERALA NAVARA AYURVEDIC FORMULATION GI

Case Description:
An association of traditional Ayurveda practitioners in Wayanad applied for GI tag protection for 'Wayanad Red Rice Herbal Extract'.

Legal Finding:
1. Community ownership must be held by a registered association representing local harvesters and traditional physicians.
2. Individual commercial entities cannot monopolize a GI name as an exclusive private patent.
3. The specification must establish proof of origin, soil parameters, microclimate factors, and historical reference in classical Ayurvedic literature dating back at least 100 years."""
    },

    # 5. Trademarks - Ayurvedic Brand & Trademark Guidance
    {
        "document_id": "SYN-TM-001",
        "title": "Trademark Registration Guidelines for Ayurvedic Brands and Medicine Names",
        "authority": "IP-SAKTI Synthetic Trademark Registry",
        "jurisdiction": "India",
        "domain": "Trademarks",
        "document_type": "Guidance",
        "effective_date": "2025-03-01",
        "source_url": "synthetic://tm-registry/ayurvedic-brand-guidelines-2025",
        "source_type": "synthetic",
        "version": "1.0",
        "content": """TRADEMARK CLASSIFICATION & DESCRIPTIVE NAMES IN AYURVEDA

Section 1. Descriptive Classical Names Prohibition (Section 9)
Under Section 9(1)(b) of the Trade Marks Act, 1999, trademarks which consist exclusively of marks or indications designating the kind, quality, or classical name of Ayurvedic medicines (e.g., 'Chyawanprash', 'Triphala Churna', 'Trikatu', 'Ashwagandharishta') are generic and descriptive, and cannot be registered as exclusive trademarks by any single company.

Section 2. Registrable Composite Marks
A manufacturer may register a composite trademark or brand name that incorporates a descriptive classical Ayurvedic term only if:
(a) The mark includes a distinct, non-descriptive brand prefix (e.g., 'KILO-Chyawanprash Special').
(b) A disclaimer is submitted disclaiming exclusive rights to the generic classical term itself under Section 17 of the Act.

Section 3. Deceptive Similarity & Public Health
Trademarks for Ayurvedic health supplements that misleadingly suggest miraculous cure properties or deceive the public regarding therapeutic class (Class 5 Pharmaceuticals vs Class 30 Food/Dietary) shall be refused registration under Section 9(2)(a)."""
    },

    # 6. Copyright - Ayurveda Documentation & Classical Texts
    {
        "document_id": "SYN-CPR-001",
        "title": "Copyright Ownership in Digitized Ayurvedic Texts and Databases",
        "authority": "IP-SAKTI Synthetic Copyright Office",
        "jurisdiction": "India",
        "domain": "Copyright",
        "document_type": "Guidance",
        "effective_date": "2024-11-12",
        "source_url": "synthetic://copyright-office/ayurveda-digitization-2024",
        "source_type": "synthetic",
        "version": "1.0",
        "content": """COPYRIGHT IN ANCIENT KNOWLEDGE & MODERN TRANSLATIONS

Section 1. Public Domain Status of Ancient Texts
Classical Ayurvedic scriptures including Charaka Samhita, Sushruta Samhita, and Bhaisajya Ratnavali are ancient public domain texts. No copyright exists in the original Sanskrit verses or traditional medicinal formulations.

Section 2. Copyright in Original Translations and Compilations
Copyright under Section 13 of the Copyright Act, 1957 subsists in:
(a) Original modern language translations (e.g. English, Hindi, Tamil translations) where skill and labor are involved.
(b) Original digital databases, annotated compilations, and structural software representations such as the Traditional Knowledge Digital Library (TKDL) layout.

Section 3. Fair Dealing for Research & Traditional Practitioners
Extracting classical recipes from modern compilations for non-commercial academic research, medical practice, or educational purposes constitutes fair dealing under Section 52(1)(a)."""
    },

    # 7. Industrial Designs - Ayurvedic Packaging & Applicator Designs
    {
        "document_id": "SYN-DES-001",
        "title": "Design Protection for Herbal Product Container and Delivery Mechanisms",
        "authority": "IP-SAKTI Synthetic Designs Office",
        "jurisdiction": "India",
        "domain": "Designs",
        "document_type": "Guidance",
        "effective_date": "2025-05-18",
        "source_url": "synthetic://designs-office/herbal-packaging-2025",
        "source_type": "synthetic",
        "version": "1.0",
        "content": """INDUSTRIAL DESIGN PROTECTION FOR AYURVEDIC PACKAGING

Section 1. Novelty in Packaging Aesthetics
Under the Designs Act, 2000, industrial design registration protects only the visual aesthetic features (shape, configuration, pattern, ornament, or color composition) applied to an article of manufacture, such as a novel Ayurvedic oil applicator bottle or ergonomic herbal pill dispenser.

Section 2. Exclusion of Functional Features
Design protection does not extend to functional mechanisms, therapeutic modes of operation, or medicinal compositions. Functional utility must be sought under Patent law, not Design law.

Section 3. Prior Publication Bar
A packaging design shall not be registered if it has been disclosed to the public in India or abroad prior to filing date, or if it lacks novelty over traditional clay pot or wooden bottle shapes documented in historical archives."""
    },

    # 8. Traditional Knowledge - TKDL Prior Art Scenario & Protection
    {
        "document_id": "SYN-TK-001",
        "title": "Traditional Knowledge Protection and TKDL Prior Art Verification Guidelines",
        "authority": "IP-SAKTI Synthetic Traditional Knowledge Authority",
        "jurisdiction": "India",
        "domain": "Traditional Knowledge",
        "document_type": "Act",
        "effective_date": "2025-08-01",
        "source_url": "synthetic://tk-authority/protection-guidelines-2025",
        "source_type": "synthetic",
        "version": "1.0",
        "content": """FRAMEWORK FOR PROTECTION OF TRADITIONAL KNOWLEDGE (TK)

Section 1. Definition of Traditional Knowledge
Traditional Knowledge encompasses indigenous wisdom, innovations, practices, and medicinal formulation knowledge passed down across generations within traditional practitioner communities in India.

Section 2. Preventive TK Protection & TKDL Search
To prevent bio-piracy and wrongful patenting of traditional Indian medicine abroad or domestically:
(a) The Traditional Knowledge Digital Library (TKDL) converts ancient Sanskrit, Arabic, Persian, Tamil, and Unani texts into international patent classification (IPC) standards.
(b) Any patent claim matching TKDL prior art entries for the same therapeutic indication shall be deemed anticipated and unpatentable.

Section 3. Defensive vs Positive TK Protection
Positive protection grants local communities sovereign control and benefit-sharing rights over commercial utilization of local medicinal formulations, while defensive protection uses TKDL to invalidate illegal foreign patent claims."""
    },

    # 9. Biodiversity / ABS - National Biodiversity Act & Benefit Sharing
    {
        "document_id": "SYN-ABS-001",
        "title": "Access and Benefit Sharing (ABS) Compliance under Biological Diversity Act",
        "authority": "IP-SAKTI Synthetic National Biodiversity Authority",
        "jurisdiction": "India",
        "domain": "Biodiversity/ABS",
        "document_type": "Act",
        "effective_date": "2024-10-01",
        "source_url": "synthetic://nba-india/abs-compliance-rules-2024",
        "source_type": "synthetic",
        "version": "1.0",
        "content": """ACCESS AND BENEFIT SHARING (ABS) REGULATORY COMPLIANCE

Section 1. Mandatory NBA Approval for Commercial Utilization
Under Section 3 and Section 4 of the Biological Diversity Act, 2002 (as amended):
(a) Foreign entities, NRI entities, or Indian companies with foreign equity sourcing Indian biological resources (e.g. Neem, Ashwagandha, Tulsi, Brahmi) for commercial research or patent application must obtain prior approval from the National Biodiversity Authority (NBA).
(b) Indian citizens and domestic entities must notify State Biodiversity Boards (SBB) prior to commercial access.

Section 2. ABS Payment Obligations
Commercial utilization of biological resources attracts mandatory Access and Benefit Sharing (ABS) fees ranging from 0.1% to 0.5% of annual gross ex-factory sales value, payable to the NBA or relevant Benefit Claimants.

Section 3. Penalties for Non-Compliance
Accessing biological resources without prior NBA approval or transferring research results to foreign entities without consent is a punishable offense under Section 55 with imprisonment and fine."""
    },

    # 10. Biodiversity / ABS - Biological Resource Access Scenario
    {
        "document_id": "SYN-ABS-002",
        "title": "Scenario Analysis: Biological Resource Access and Foreign Patent Approval",
        "authority": "IP-SAKTI Synthetic National Biodiversity Authority",
        "jurisdiction": "India",
        "domain": "Biodiversity/ABS",
        "document_type": "Scenario",
        "effective_date": "2026-01-10",
        "source_url": "synthetic://nba-india/resource-access-scenario-2026",
        "source_type": "synthetic",
        "version": "1.0",
        "content": """CASE SCENARIO: INTERNATIONAL PATENT FILING USING INDIAN HERBAL EXTRACTS

Fact Pattern:
A joint venture company based in Bengaluru with 15% foreign venture capital equity isolated an active alkaloid from Indian Sarpagandha (Rauvolfia serpentina) and filed a PCT patent application without NBA approval.

Legal Consequences & Mandatory Remedies:
1. Filing a patent application inside or outside India based on Indian biological resources without NBA approval violates Section 6 of the Biological Diversity Act.
2. The Indian Patent Office shall withhold final patent grant until an official Form III NBA clearance certificate is produced.
3. The applicant must execute an ABS agreement committing 0.25% of commercial revenues to local tribal conservation communities in the Western Ghats."""
    },

    # 11. Drug Classification - Classical Ayurvedic Medicine
    {
        "document_id": "SYN-DRG-001",
        "title": "Regulatory Standard for Classical Ayurvedic Medicine Classification",
        "authority": "IP-SAKTI Synthetic AYUSH Drug Controller",
        "jurisdiction": "India",
        "domain": "Drug Classification",
        "document_type": "Rule",
        "effective_date": "2025-04-01",
        "source_url": "synthetic://ayush-controller/classical-medicine-rules-2025",
        "source_type": "synthetic",
        "version": "1.0",
        "content": """DRUGS AND COSMETICS ACT: CLASSICAL AYURVEDIC MEDICINES

Section 1. Definition of Classical Medicine
Under Section 3(a) of the Drugs and Cosmetics Act, 1940, a Classical Ayurvedic Medicine is defined as a formulation manufactured strictly according to the authoritative books and recipe texts specified in the First Schedule of the Act (including Ayurvedic Pharmacopoeia of India, Charaka Samhita, Sharangdhara Samhita).

Section 2. Manufacturing Licensing Requirements
Classical medicines do not require clinical trial safety trials for initial manufacturing license, provided manufacturing adheres strictly to Good Manufacturing Practices (GMP) under Schedule T and specified textual ingredients.

Section 3. Labeling and Claim Restrictions
Classical medicines must display classical scriptural reference name and cannot claim instant miraculous cure rights prohibited under the Drugs and Magic Remedies (Objectionable Advertisements) Act, 1954."""
    },

    # 12. Drug Classification - Proprietary Ayurvedic Medicine
    {
        "document_id": "SYN-DRG-002",
        "title": "Regulatory Framework for Patent or Proprietary Ayurvedic Medicines",
        "authority": "IP-SAKTI Synthetic AYUSH Drug Controller",
        "jurisdiction": "India",
        "domain": "Drug Classification",
        "document_type": "Rule",
        "effective_date": "2025-04-15",
        "source_url": "synthetic://ayush-controller/proprietary-medicine-rules-2025",
        "source_type": "synthetic",
        "version": "1.0",
        "content": """PATENT OR PROPRIETARY AYURVEDIC MEDICINE REGULATION

Section 1. Definition
Under Section 3(h) of the Drugs and Cosmetics Act, 1940, a 'Patent or Proprietary Medicine' in relation to ASU (Ayurveda, Siddha, Unani) systems means a formulation containing ingredients specified in the classical texts, but which is NOT a classical formulation listed in the First Schedule, or which is prepared using novel modern excipients and dosages.

Section 2. Safety and Efficacy Proof
Applicants for a Proprietary Ayurvedic Medicine license must submit:
(a) Scientific rationale for ingredient combination.
(b) Safety toxicological data where non-classical extraction solvents (e.g. ethanol/hydro-alcoholic extracts exceeding classical water decoction) are utilized.
(c) Proof that ingredients do not include Schedule E(1) poisonous plants above permissible limits."""
    },

    # 13. Drug Classification - Phytopharmaceuticals vs Classical vs Food
    {
        "document_id": "SYN-DRG-003",
        "title": "Comparative Matrix for Phytopharmaceutical, Ayurveda-Aahar, and Cosmetic Classification",
        "authority": "IP-SAKTI Synthetic AYUSH & FSSAI Joint Licensing Authority",
        "jurisdiction": "India",
        "domain": "Drug Classification",
        "document_type": "Guidance",
        "effective_date": "2025-07-01",
        "source_url": "synthetic://ayush-fssai/drug-food-classification-2025",
        "source_type": "synthetic",
        "version": "1.0",
        "content": """INTER-CATEGORY CLASSIFICATION MATRIX FOR AYURVEDIC PRODUCTS

Category 1: Phytopharmaceutical Drug
- Governed by Central Drugs Standard Control Organization (CDSCO) under Drugs & Cosmetics Rules.
- Requires standardized fraction, chemical marker identification, phase I-III clinical trial safety data.

Category 2: Ayurveda-Aahar (Ayurvedic Food Product)
- Governed by FSSAI under Food Safety and Standards (Ayurveda Aahar) Regulations, 2022.
- Intended strictly for dietary consumption and health maintenance, NOT for therapeutic cure of disease.
- Excludes Schedule E(1) toxic plants and pure synthetic chemical additions.

Category 3: Ayurvedic Cosmetic
- Topically applied formulations for beautifying or cleansing. Cannot claim systemic medicinal healing.

Category 4: Unknown / Insufficient Information
- Formulations lacking clear ingredient specification or intended mode of administration shall be quarantined as 'Unknown/Insufficient' until verified by the Expert Classification Committee."""
    },

    # 14. International - TRIPS Agreement & IP Standards
    {
        "document_id": "SYN-INT-001",
        "title": "TRIPS Agreement Compliance and Traditional Knowledge Exceptions",
        "authority": "IP-SAKTI Synthetic WIPO & WTO Desk",
        "jurisdiction": "International",
        "domain": "Patents",
        "document_type": "Treaty",
        "effective_date": "2024-01-01",
        "source_url": "synthetic://wto-wipo/trips-tk-exceptions-2024",
        "source_type": "synthetic",
        "version": "1.0",
        "content": """WTO TRIPS AGREEMENT & TRADITIONAL KNOWLEDGE PROVISIONS

Article 27. Patentable Subject Matter
Members shall provide patents for any inventions, whether products or processes, in all fields of technology, provided they are new, involve an inventive step, and are capable of industrial application.

Article 27.3(b). Plant Variety and Biological Exclusions
Members may exclude from patentability plants and animals other than micro-organisms, and essentially biological processes for the production of plants or animals. Members shall provide for the protection of plant varieties either by patents or by an effective sui generis system.

Article 8. Public Health & Flexibility
Members may adopt measures necessary to protect public health and nutrition, and to promote the public interest in sectors of vital importance to their socio-economic and technological development, provided such measures are consistent with TRIPS."""
    },

    # 15. International - Patent Cooperation Treaty (PCT) Route
    {
        "document_id": "SYN-INT-002",
        "title": "PCT International Patent Filing Route for Herbal Inventions",
        "authority": "IP-SAKTI Synthetic WIPO Desk",
        "jurisdiction": "International",
        "domain": "Patents",
        "document_type": "Treaty",
        "effective_date": "2024-05-10",
        "source_url": "synthetic://wipo/pct-herbal-filing-route-2024",
        "source_type": "synthetic",
        "version": "1.0",
        "content": """PATENT COOPERATION TREATY (PCT) PROCEDURES FOR AYURVEDIC FORMULATIONS

Rule 1. International Application Filing
An applicant seeking international patent protection across multiple countries may file a single PCT international application with the Receiving Office (such as Indian Patent Office or WIPO).

Rule 2. International Searching Authority (ISA) and TKDL Search
The International Searching Authority performs an International Search Report (ISR). Major ISAs (including Indian Patent Office) search the TKDL database during ISR generation to evaluate novelty against traditional Indian medicine prior art.

Rule 3. National Phase Entry (30/31 Months)
The applicant must enter national phase in target jurisdictions (e.g. USPTO, EPO, JPO) within 30 or 31 months from priority date. National patent laws (such as Section 3(p) in India) apply fully during national phase examination."""
    },

    # 16. International - Nagoya Protocol & CBD Framework
    {
        "document_id": "SYN-INT-003",
        "title": "Nagoya Protocol on Access to Genetic Resources and Fair Benefit Sharing",
        "authority": "IP-SAKTI Synthetic UN Secretariat for Biodiversity",
        "jurisdiction": "International",
        "domain": "Biodiversity/ABS",
        "document_type": "Treaty",
        "effective_date": "2024-03-15",
        "source_url": "synthetic://cbd-un/nagoya-protocol-framework-2024",
        "source_type": "synthetic",
        "version": "1.0",
        "content": """NAGOYA PROTOCOL ON ACCESS AND BENEFIT SHARING (ABS)

Article 5. Fair and Equitable Benefit-Sharing
Benefits arising from the utilization of genetic resources as well as subsequent applications and commercialization shall be shared in a fair and equitable way with the party providing such resources, upon mutually agreed terms (MAT).

Article 6. Prior Informed Consent (PIC)
Access to genetic resources for commercial or research utilization shall be subject to the Prior Informed Consent (PIC) of the party providing such resources, unless otherwise determined by that party.

Article 7. Access to Traditional Knowledge Associated with Genetic Resources
Parties shall take measures to ensure that traditional knowledge associated with genetic resources held by indigenous and local communities is accessed with their prior informed consent and involvement."""
    },

    # 17. International - WIPO GRATK Treaty 2024
    {
        "document_id": "SYN-INT-004",
        "title": "WIPO Treaty on Intellectual Property, Genetic Resources and Associated Traditional Knowledge",
        "authority": "IP-SAKTI Synthetic WIPO Secretariat",
        "jurisdiction": "International",
        "domain": "Traditional Knowledge",
        "document_type": "Treaty",
        "effective_date": "2024-05-24",
        "source_url": "synthetic://wipo/gratk-treaty-2024",
        "source_type": "synthetic",
        "version": "1.0",
        "content": """WIPO GRATK TREATY (MAY 2024)

Article 3. Mandatory Patent Disclosure Requirement
Where a claimed invention in a patent application is materially based on genetic resources or associated traditional knowledge, applicants shall disclose:
(a) The country of origin of the genetic resource, or the indigenous community providing the traditional knowledge.
(b) If country of origin is unknown, the direct source from which the applicant received the genetic resource or traditional knowledge.

Article 4. Sanctions and Legal Consequences
Contracting Parties shall provide administrative or legal remedies where an applicant intentionally fails to disclose mandatory origin details."""
    },

    # 18. Conflicting Document Version 1 - AYUSH Policy 2024 (Historical)
    {
        "document_id": "SYN-VER-2024",
        "title": "AYUSH Regulatory & Export Policy Guidelines (Version 2024 - Historical)",
        "authority": "IP-SAKTI Synthetic Ministry of AYUSH",
        "jurisdiction": "India",
        "domain": "Drug Classification",
        "document_type": "Guidance",
        "effective_date": "2024-01-01",
        "source_url": "synthetic://ayush-ministry/export-policy-2024",
        "source_type": "synthetic",
        "version": "2024.1",
        "content": """HISTORICAL POLICY 2024: AYURVEDA HERBAL EXPORT STANDARDS

Clause 1. Preliminary Export Approvals (2024 Framework)
Under the 2024 policy framework, export of raw Ayurvedic herbal decoctions required a basic SBB notification and standard phytosanitary certificate. Benefit sharing (ABS) fees for raw herbal exports were capped at 0.1% of FOB export value.

Clause 2. Heavy Metal Limits (2024 Baseline)
Permissible limits for heavy metals in exported herbal formulations were: Lead (10 ppm), Cadmium (0.3 ppm), Mercury (1.0 ppm), Arsenic (3.0 ppm).

[NOTE: THIS IS THE 2024 HISTORICAL VERSION. SUPERSEDED BY VERSION 2026 FOR CURRENT COMPLIANCE.]"""
    },

    # 19. Conflicting Document Version 2 - AYUSH Policy 2026 (Current)
    {
        "document_id": "SYN-VER-2026",
        "title": "AYUSH Regulatory & Export Policy Guidelines (Version 2026 - Current)",
        "authority": "IP-SAKTI Synthetic Ministry of AYUSH",
        "jurisdiction": "India",
        "domain": "Drug Classification",
        "document_type": "Guidance",
        "effective_date": "2026-01-01",
        "source_url": "synthetic://ayush-ministry/export-policy-2026",
        "source_type": "synthetic",
        "version": "2026.1",
        "content": """CURRENT POLICY 2026: AYURVEDA HERBAL EXPORT STANDARDS

Clause 1. Mandatory Pre-Export NBA Clearance (2026 Update)
Effective January 1, 2026, all commercial exports of raw or processed Ayurvedic biological resources require mandatory prior NBA clearance certificate (Form III). ABS fees are updated to 0.25% of annual gross export turnover.

Clause 2. Strict Heavy Metal & Pesticide Limits (2026 Standard)
Updated heavy metal safety limits for 2026 exports: Lead (5.0 ppm), Cadmium (0.1 ppm), Mercury (0.5 ppm), Arsenic (1.5 ppm). Mandatory pesticide residue testing report is required prior to export customs clearance.

[NOTE: THIS IS THE 2026 CURRENT APPLICABLE POLICY VERSION.]"""
    },

    # 20. International - Madrid System for Trademark Registration
    {
        "document_id": "SYN-INT-005",
        "title": "Madrid System International Trademark Filing for Ayurvedic Brands",
        "authority": "IP-SAKTI Synthetic WIPO Madrid Registry",
        "jurisdiction": "International",
        "domain": "Trademarks",
        "document_type": "Treaty",
        "effective_date": "2024-08-15",
        "source_url": "synthetic://wipo/madrid-system-ayurveda-2024",
        "source_type": "synthetic",
        "version": "1.0",
        "content": """MADRID SYSTEM FOR INTERNATIONAL BRAND PROTECTION

Article 2. Basic Application Requirement
An Ayurvedic brand owner registered with the Indian Trademark Registry may file a single international trademark application under the Madrid Protocol specifying multiple member countries.

Article 3. Dependency (Central Attack)
For 5 years from international registration date, protection remains dependent on the basic Indian trademark. If the Indian mark is canceled, the international registration is invalidated across all designated countries."""
    },

    # 21. Specific Keyword Test Case - Section 3(p) Deep Dive
    {
        "document_id": "SYN-PAT-003",
        "title": "Legal Precedents on Section 3(p) and Traditional Knowledge Invalidation",
        "authority": "IP-SAKTI Synthetic Patent Appellate Board",
        "jurisdiction": "India",
        "domain": "Patents",
        "document_type": "Guidance",
        "effective_date": "2025-11-05",
        "source_url": "synthetic://patent-board/section-3p-precedents-2025",
        "source_type": "synthetic",
        "version": "1.0",
        "content": """ANALYSIS OF SECTION 3(p) OBJECTIONS IN AYURVEDA PATENTS

Key Holding:
Where an applicant claims a formulation containing Curcuma longa (Turmeric) and Zingiber officinale (Ginger) for anti-inflammatory treatment, Section 3(p) operates as an absolute statutory bar if the medicinal properties of both plants for inflammatory disorders are documented in TKDL or Ayurvedic text Charaka Samhita.

To overcome Section 3(p), the specification must establish:
1. Synergistic interaction ratio which is non-obvious.
2. Novel extraction medium yielding a distinct non-obvious bio-active composition.
3. Comparative experimental evidence showing superior therapeutic outcome over conventional classical decoction."""
    },

    # 22. Specific Keyword Test Case - Budapest Treaty Microorganisms
    {
        "document_id": "SYN-INT-006",
        "title": "Budapest Treaty Deposit Requirements for Herbal Fermentation Microorganisms",
        "authority": "IP-SAKTI Synthetic WIPO Depositary",
        "jurisdiction": "International",
        "domain": "Patents",
        "document_type": "Treaty",
        "effective_date": "2024-04-12",
        "source_url": "synthetic://wipo/budapest-treaty-microorganisms-2024",
        "source_type": "synthetic",
        "version": "1.0",
        "content": """BUDAPEST TREATY ON INTERNATIONAL DEPOSIT OF MICROORGANISMS

Rule 1. Mandatory Biological Deposit
When an Ayurvedic patent process involves a novel fermentation strain (e.g. Asava or Arishta yeast starter cultures) that cannot be described in written text, the strain must be deposited in an International Depositary Authority (IDA) under the Budapest Treaty.

Rule 2. Deposit Reference in Patent Specification
The deposit receipt number, strain access code, and depositary location must be disclosed in the patent specification before publication."""
    },

    # 23. Hague System - Industrial Packaging International
    {
        "document_id": "SYN-INT-007",
        "title": "Hague Agreement Concerning International Registration of Industrial Designs",
        "authority": "IP-SAKTI Synthetic WIPO Hague Registry",
        "jurisdiction": "International",
        "domain": "Designs",
        "document_type": "Treaty",
        "effective_date": "2024-07-01",
        "source_url": "synthetic://wipo/hague-system-designs-2024",
        "source_type": "synthetic",
        "version": "1.0",
        "content": """HAGUE SYSTEM FOR INTERNATIONAL INDUSTRIAL DESIGNS

Article 1. Single Filing Mechanism
Designers of novel Ayurvedic container bottles, applicators, and eco-friendly packaging can file a single international design application under the Hague Agreement to secure design protection in up to 90 member territories.

Article 2. Examination by Contracting Parties
Each designated country reviews the application under its domestic design law for novelty and non-functionality within 6 to 12 months."""
    },

    # 24. Traditional Knowledge - Sui Generis Protection Model
    {
        "document_id": "SYN-TK-002",
        "title": "Sui Generis Legal Protection Framework for Traditional Medicinal Knowledge",
        "authority": "IP-SAKTI Synthetic Traditional Knowledge Authority",
        "jurisdiction": "India",
        "domain": "Traditional Knowledge",
        "document_type": "Guidance",
        "effective_date": "2026-01-20",
        "source_url": "synthetic://tk-authority/sui-generis-framework-2026",
        "source_type": "synthetic",
        "version": "1.0",
        "content": """SUI GENERIS LEGISLATION FOR COMMUNITY TRADITIONAL KNOWLEDGE

Section 1. Recognition of Indigenous Community Rights
Traditional knowledge held by indigenous healers, Vaidyas, and tribal communities is recognized as collective intellectual property belonging to the community rather than private individuals.

Section 2. Royalty and Benefit Sharing Registry
Commercial entities using community traditional knowledge must register licensing agreements with the National TK Registry and deposit 1% of gross revenue into the Community Biodiversity Fund."""
    },

    # 25. Drug Classification - Phytopharmaceuticals Detailed Regulation
    {
        "document_id": "SYN-DRG-004",
        "title": "CDSCO Regulatory Guidelines for Phytopharmaceutical Drug Approval",
        "authority": "IP-SAKTI Synthetic CDSCO",
        "jurisdiction": "India",
        "domain": "Drug Classification",
        "document_type": "Rule",
        "effective_date": "2025-10-10",
        "source_url": "synthetic://cdsco/phytopharmaceutical-rules-2025",
        "source_type": "synthetic",
        "version": "1.0",
        "content": """CDSCO REGULATION OF PHYTOPHARMACEUTICAL DRUGS

Section 1. Submission Requirements (Appendix XIII)
An application for manufacturing or importing a Phytopharmaceutical drug must include:
1. High-Performance Liquid Chromatography (HPLC) / Mass Spectrometry fingerprinting of 4 active chemical markers.
2. Acute and sub-acute animal toxicity data.
3. Human Phase I safety and Phase II/III clinical trial protocols approved by the Central Licensing Authority."""
    },

    # 26. Patents - Prior Art Search & Patentability Criteria
    {
        "document_id": "SYN-PAT-004",
        "title": "Manual of Patent Office Practice and Procedure: Herbal Inventions",
        "authority": "IP-SAKTI Synthetic Patent Office",
        "jurisdiction": "India",
        "domain": "Patents",
        "document_type": "Guidance",
        "effective_date": "2025-06-30",
        "source_url": "synthetic://patent-office/manual-herbal-practice-2025",
        "source_type": "synthetic",
        "version": "1.0",
        "content": """EXAMINATION MANUAL: HERBAL AND BOTANICAL PATENT CLAIMS

Section 1. Novelty Assessment
A herbal composition claim lacks novelty if all ingredients and therapeutic applications are disclosed in any single prior art document, classical text, or public database (TKDL, PubMed, Google Patents).

Section 2. Inventive Step (Section 2(1)(j))
An inventive step requires a technical advance or economic significance that is not obvious to a person skilled in the art of herbal formulation and pharmacology.

Section 3. Section 3(h) Agriculture and Horticulture Exclusion
Methods of cultivation or harvesting of medicinal plants (e.g. methods of growing organic Ashwagandha) are excluded from patentability under Section 3(h)."""
    },

    # 27. Trademarks - Certification Marks & Collective Marks for Herbal Products
    {
        "document_id": "SYN-TM-002",
        "title": "Certification Marks and Collective Marks for Certified Ayurvedic Quality",
        "authority": "IP-SAKTI Synthetic Trademark Registry",
        "jurisdiction": "India",
        "domain": "Trademarks",
        "document_type": "Rule",
        "effective_date": "2024-09-05",
        "source_url": "synthetic://tm-registry/certification-marks-ayurveda-2024",
        "source_type": "synthetic",
        "version": "1.0",
        "content": """CERTIFICATION MARKS AND AYUSH MARK REGULATION

Section 1. AYUSH Premium Mark Certification
The AYUSH Mark is a certified quality mark issued to herbal drug manufacturers complying with WHO Good Manufacturing Practices (GMP) and heavy metal residue limits.

Section 2. Collective Trademarks for Cooperative Societies
Associations of traditional herbal harvesters may register collective marks under Section 61 of the Trade Marks Act, 1999 to distinguish cooperative products from commercial counterfeits."""
    },

    # 28. Copyright - Traditional Knowledge Digital Library Access Terms
    {
        "document_id": "SYN-CPR-002",
        "title": "TKDL Access Agreement and Intellectual Property Conditions",
        "authority": "IP-SAKTI Synthetic CSIR & TKDL Unit",
        "jurisdiction": "India",
        "domain": "Copyright",
        "document_type": "Rule",
        "effective_date": "2025-02-14",
        "source_url": "synthetic://csir-tkdl/access-agreement-2025",
        "source_type": "synthetic",
        "version": "1.0",
        "content": """TRADITIONAL KNOWLEDGE DIGITAL LIBRARY (TKDL) ACCESS TERMS

Clause 1. Confidential Access for Patent Offices
International Patent Offices (such as EPO, USPTO, JPO, IPO) access TKDL under non-disclosure agreements strictly for patent examination and prior art verification purposes.

Clause 2. Prohibition on Public Redistribution
The TKDL database structure, translation glossaries, and digitised IPC index codes are protected under Indian Copyright law. Third-party commercial scraping or public distribution without CSIR authorization is strictly prohibited."""
    },

    # 29. Biodiversity / ABS - State Biodiversity Board Regulations
    {
        "document_id": "SYN-ABS-003",
        "title": "State Biodiversity Board Intimation and Revenue Sharing Regulations",
        "authority": "IP-SAKTI Synthetic State Biodiversity Board",
        "jurisdiction": "India",
        "domain": "Biodiversity/ABS",
        "document_type": "Rule",
        "effective_date": "2024-12-01",
        "source_url": "synthetic://sbb-state/revenue-sharing-rules-2024",
        "source_type": "synthetic",
        "version": "1.0",
        "content": """STATE BIODIVERSITY BOARD (SBB) COMMERCIAL ACCESS RULES

Section 1. Obligation of Domestic Manufacturers
Indian manufacturers acquiring biological resources from local farmers or forests for commercial Ayurvedic drug production must give prior intimation to the State Biodiversity Board in Form I.

Section 2. Allocation of ABS Funds
Of the benefit sharing funds collected by SBB, 95% shall be disbursed directly to local Biodiversity Management Committees (BMCs) and benefit claimants for conservation and habitat protection."""
    },

    # 30. Irrelevant / Distractor Document - Synthetic Medical Device Regulations
    {
        "document_id": "SYN-DIS-001",
        "title": "Medical Devices Quality Management System Standard (Distractor Document)",
        "authority": "IP-SAKTI Synthetic Medical Device Bureau",
        "jurisdiction": "India",
        "domain": "Drug Classification",
        "document_type": "Rule",
        "effective_date": "2025-05-01",
        "source_url": "synthetic://med-device/qms-standard-2025",
        "source_type": "synthetic",
        "version": "1.0",
        "content": """MEDICAL DEVICE QUALITY STANDARDS (NON-HERBAL DISTRACTOR)

Section 1. Class C Electro-Mechanical Surgical Scopes
This document specifies calibration standards for X-ray equipment, MRI imaging software, and titanium surgical orthopedic implants. It does NOT cover Ayurvedic herbal preparations, biological resources, or patentability of traditional knowledge."""
    },

    # 31. Irrelevant / Distractor Document - Synthetic Telecom Spectrum Licensing
    {
        "document_id": "SYN-DIS-002",
        "title": "5G Telecommunications Spectrum Licensing Framework (Distractor Document)",
        "authority": "IP-SAKTI Synthetic Telecom Authority",
        "jurisdiction": "India",
        "domain": "Patents",
        "document_type": "Act",
        "effective_date": "2024-04-01",
        "source_url": "synthetic://telecom/5g-spectrum-2024",
        "source_type": "synthetic",
        "version": "1.0",
        "content": """TELECOM SPECTRUM ALLOCATION GUIDELINES (NON-HERBAL DISTRACTOR)

Section 1. Frequency Band Auction Parameters
This spectrum allocation guideline applies to 3.5 GHz cellular towers and wireless radio frequency distribution. This document is unrelated to botanical medicine, traditional knowledge, or biological diversity."""
    },

    # 32. Drug Classification - Cosmetic vs Ayurveda Borderline Case
    {
        "document_id": "SYN-DRG-005",
        "title": "Borderline Herbal Products: Cosmetic vs Ayurvedic Drug Criteria",
        "authority": "IP-SAKTI Synthetic AYUSH Licensing Committee",
        "jurisdiction": "India",
        "domain": "Drug Classification",
        "document_type": "Guidance",
        "effective_date": "2026-02-15",
        "source_url": "synthetic://ayush-licensing/cosmetic-vs-drug-2026",
        "source_type": "synthetic",
        "version": "1.0",
        "content": """BORDERLINE PRODUCT EVALUATION: AYURVEDIC MEDICINE VS COSMETIC

Section 1. Primary Intended Use Criterion
A herbal skin cream containing Neem and Aloe Vera is classified as an Ayurvedic Medicine if its primary claim is therapeutic cure of skin diseases (e.g. eczema, psoriasis) supported by classical texts.

Section 2. Cosmetic Classification
If the same formulation is advertised solely for moisturizing, improving skin appearance, or cleansing, it must be licensed as a Cosmetic under Schedule Q of Drugs & Cosmetics Rules, and cannot make therapeutic medical claims."""
    }
]


def generate():
    print(f"Generating {len(DOCUMENTS)} synthetic documents in {SYNTHETIC_DIR}...")
    for doc in DOCUMENTS:
        # Save JSON file
        json_path = os.path.join(SYNTHETIC_DIR, f"{doc['document_id']}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2)
        
        # Save readable TXT file
        txt_path = os.path.join(SYNTHETIC_DIR, f"{doc['document_id']}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"TITLE: {doc['title']}\n")
            f.write(f"AUTHORITY: {doc['authority']}\n")
            f.write(f"JURISDICTION: {doc['jurisdiction']}\n")
            f.write(f"DOMAIN: {doc['domain']}\n")
            f.write(f"DOCUMENT_TYPE: {doc['document_type']}\n")
            f.write(f"EFFECTIVE_DATE: {doc['effective_date']}\n")
            f.write(f"SOURCE_URL: {doc['source_url']}\n")
            f.write(f"SOURCE_TYPE: {doc['source_type']}\n")
            f.write(f"VERSION: {doc['version']}\n")
            f.write("\n--- CONTENT ---\n")
            f.write(doc['content'])
            
    print(f"Successfully generated {len(DOCUMENTS)} synthetic corpus files.")

if __name__ == "__main__":
    generate()
