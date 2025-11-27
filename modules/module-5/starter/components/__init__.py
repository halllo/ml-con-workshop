"""
Kubeflow Pipeline Components
=============================

This package contains individual pipeline components for the movie recommendation system.

Components:
- data_prep: Download and prepare MovieLens dataset
- train: Train collaborative filtering model
- evaluate: Evaluate model performance
- deploy: Deploy model to KServe (optional)
"""

__all__ = ['data_prep', 'train', 'evaluate']
