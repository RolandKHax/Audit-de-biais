# Fichier: tests/__init__.py
"""Tests package initialization"""


if __name__ == "__main__":
    # Exécuter les tests
    import pytest
    pytest.main([__file__, "-v", "--cov=src", "--cov-report=html"])