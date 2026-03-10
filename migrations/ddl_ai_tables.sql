-- ═══════════════════════════════════════════════════════════════
--  PCD-AI Service — Database DDL
--  Panel Change Detection AI Tables
--  Prefix: ai_
--  Panel identity = location_id + taskcheck_id
-- ═══════════════════════════════════════════════════════════════

-- ─── ai_baselines ────────────────────────────────────────────
-- One row per reference photo. One baseline can have many comparisons.
-- ─────────────────────────────────────────────────────────────

CREATE TABLE ai_baselines (
    id                  SERIAL PRIMARY KEY,
    location_id         VARCHAR(50) NOT NULL,
    taskcheck_id        INTEGER NOT NULL,

    -- Image storage
    image_url           TEXT NOT NULL,
    enhanced_cache_path VARCHAR(500),       -- CLAHE preprocessed grayscale path
    color_cache_path    VARCHAR(500),       -- Color-preserved copy path

    -- Quality metrics (computed on registration)
    blur_score          FLOAT,
    brightness          FLOAT,
    quality_warning     BOOLEAN DEFAULT FALSE,

    -- Lifecycle
    is_active           BOOLEAN DEFAULT TRUE,
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX idx_ai_baselines_location_id ON ai_baselines (location_id);
CREATE INDEX idx_ai_baselines_taskcheck_id ON ai_baselines (taskcheck_id);
CREATE INDEX idx_ai_baselines_loc_task_active ON ai_baselines (location_id, taskcheck_id, is_active);


-- ─── ai_comparisons ──────────────────────────────────────────
-- One row per patrol comparison. Many comparisons per baseline.
-- ─────────────────────────────────────────────────────────────

CREATE TABLE ai_comparisons (
    id                  SERIAL PRIMARY KEY,

    -- Relationship
    baseline_id         INTEGER NOT NULL REFERENCES ai_baselines(id),

    -- Input
    patrol_image_url    TEXT NOT NULL,

    -- Results
    ratio               FLOAT,              -- SVM probability (change %)
    similarity_percent  FLOAT,              -- (1 - ratio) * 100
    matching            BOOLEAN,            -- ratio < threshold = matching (no change)
    status              VARCHAR(30) DEFAULT 'PENDING',
                                            -- PENDING | PROCESSING | COMPLETED | FAILED | FRAUD

    -- Fraud Detection (M2 - CLIP)
    is_valid            BOOLEAN,            -- CLIP fraud check passed
    fraud_score         FLOAT,              -- CLIP cosine similarity

    -- Structure Features (M4 - Grayscale Path)
    ssim_score          FLOAT,
    ms_ssim_score       FLOAT,
    edge_score          FLOAT,
    histogram_score     FLOAT,
    cluster_score       FLOAT,
    max_diff_area       FLOAT,              -- Most important feature (0.87 weight)
    stability_score     FLOAT,

    -- Color Features (M4 - Color Path)
    max_hue_shift       FLOAT,              -- HSV hue delta
    max_delta_e         FLOAT,              -- LAB perceptual color distance

    -- Alignment (M3)
    alignment_inliers   INTEGER,
    alignment_method    VARCHAR(20),        -- lightglue | orb

    -- Quality & Output
    blur_score          FLOAT,
    heatmap_path        VARCHAR(500),
    processing_ms       INTEGER,
    error_message       TEXT,

    -- Supervisor Feedback
    supervisor_action   VARCHAR(30),        -- CONFIRM_CHANGE | REJECT_CHANGE | CONFIRM_NORMAL
    supervisor_notes    TEXT,
    supervisor_reviewed_at TIMESTAMP,

    -- Timestamps
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at        TIMESTAMP
);

-- Indexes
CREATE INDEX idx_ai_comparisons_baseline_id ON ai_comparisons (baseline_id);
CREATE INDEX idx_ai_comparisons_status ON ai_comparisons (status);
CREATE INDEX idx_ai_comparisons_baseline_created ON ai_comparisons (baseline_id, created_at);
CREATE INDEX idx_ai_comparisons_feedback ON ai_comparisons (supervisor_action);
