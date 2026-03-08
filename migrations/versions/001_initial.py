"""create ai_baselines and ai_comparisons tables

Revision ID: 001
Revises: -
Create Date: 2026-03-05

"""
from alembic import op
import sqlalchemy as sa

revision = '001_initial'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── ai_baselines ──
    op.create_table(
        'ai_baselines',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('processing_id', sa.String(50), nullable=False),
        sa.Column('panel_id', sa.String(50), nullable=False),
        sa.Column('site_id', sa.String(50), nullable=True),
        sa.Column('location_id', sa.String(50), nullable=True),
        sa.Column('image_url', sa.Text(), nullable=False),
        sa.Column('enhanced_cache_path', sa.String(500), nullable=True),
        sa.Column('color_cache_path', sa.String(500), nullable=True),
        sa.Column('blur_score', sa.Float(), nullable=True),
        sa.Column('brightness', sa.Float(), nullable=True),
        sa.Column('quality_warning', sa.Boolean(), server_default='false'),
        sa.Column('is_active', sa.Boolean(), server_default='true'),
        sa.Column('replaced_by', sa.String(50), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('processing_id'),
    )
    op.create_index('idx_ai_baselines_panel_id', 'ai_baselines', ['panel_id'])
    op.create_index('idx_ai_baselines_processing_id', 'ai_baselines', ['processing_id'])
    op.create_index('idx_ai_baselines_panel_active', 'ai_baselines', ['panel_id', 'is_active'])

    # ── ai_comparisons ──
    op.create_table(
        'ai_comparisons',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('definition_id', sa.Integer(), nullable=True),
        sa.Column('execution_id', sa.Integer(), nullable=True),
        sa.Column('baseline_id', sa.Integer(), nullable=False),
        sa.Column('patrol_image_url', sa.Text(), nullable=False),
        sa.Column('ratio', sa.Float(), nullable=True),
        sa.Column('matching', sa.Boolean(), nullable=True),
        sa.Column('status', sa.String(30), server_default='PENDING'),
        sa.Column('is_valid', sa.Boolean(), nullable=True),
        sa.Column('fraud_score', sa.Float(), nullable=True),
        sa.Column('ssim_score', sa.Float(), nullable=True),
        sa.Column('ms_ssim_score', sa.Float(), nullable=True),
        sa.Column('edge_score', sa.Float(), nullable=True),
        sa.Column('histogram_score', sa.Float(), nullable=True),
        sa.Column('cluster_score', sa.Float(), nullable=True),
        sa.Column('max_diff_area', sa.Float(), nullable=True),
        sa.Column('stability_score', sa.Float(), nullable=True),
        sa.Column('max_hue_shift', sa.Float(), nullable=True),
        sa.Column('max_delta_e', sa.Float(), nullable=True),
        sa.Column('alignment_inliers', sa.Integer(), nullable=True),
        sa.Column('alignment_method', sa.String(20), nullable=True),
        sa.Column('blur_score', sa.Float(), nullable=True),
        sa.Column('heatmap_path', sa.String(500), nullable=True),
        sa.Column('processing_ms', sa.Integer(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('supervisor_action', sa.String(30), nullable=True),
        sa.Column('supervisor_notes', sa.Text(), nullable=True),
        sa.Column('supervisor_reviewed_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['baseline_id'], ['ai_baselines.id']),
    )
    op.create_index('idx_ai_comparisons_baseline_id', 'ai_comparisons', ['baseline_id'])
    op.create_index('idx_ai_comparisons_status', 'ai_comparisons', ['status'])
    op.create_index('idx_ai_comparisons_baseline_created', 'ai_comparisons', ['baseline_id', 'created_at'])
    op.create_index('idx_ai_comparisons_feedback', 'ai_comparisons', ['supervisor_action'])


def downgrade() -> None:
    op.drop_table('ai_comparisons')
    op.drop_table('ai_baselines')
