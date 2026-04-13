#!/usr/bin/env python3
"""
Test script for multi-relation extraction
Demonstrates usage and validates the framework
"""

import os
from multi_relation_kg_builder import MultiRelationKGBuilder
from relation_extraction_config import list_available_relations, get_relation_config


def test_list_relations():
    """Test listing available relations"""
    print("="*60)
    print("TEST: List Available Relations")
    print("="*60)
    
   