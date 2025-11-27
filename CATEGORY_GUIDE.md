# Amazon Reviews Dataset - Category Selection Guide

## 🎯 Recommended Categories (Easiest to Start)

### **Best Choice: All_Beauty** ⭐
- **Reviews**: 701.5K
- **Users**: 632.0K  
- **Items**: 112.6K
- **File Size**: ~31.6M tokens (relatively small)
- **Why**: Smallest category, perfect for testing and development
- **Download Links**:
  - Review: `review_All_Beauty.jsonl.gz`
  - Metadata: `meta_All_Beauty.jsonl.gz`

### **Second Choice: Baby_Products**
- **Reviews**: 6.0M
- **Users**: 3.4M
- **Items**: 217.7K
- **Why**: Still manageable, good for more comprehensive testing

### **Third Choice: Appliances**
- **Reviews**: 2.1M
- **Users**: 1.8M
- **Items**: 94.3K
- **Why**: Small item count, moderate review count

## 📊 Category Size Comparison

| Category | Reviews | Users | Items | Difficulty |
|----------|---------|-------|-------|------------|
| **All_Beauty** | 701.5K | 632.0K | 112.6K | ⭐ Easiest |
| Appliances | 2.1M | 1.8M | 94.3K | ⭐⭐ Easy |
| Baby_Products | 6.0M | 3.4M | 217.7K | ⭐⭐ Easy |
| Amazon_Fashion | 2.5M | 2.0M | 825.9K | ⭐⭐⭐ Medium |
| Arts_Crafts_and_Sewing | 9.0M | 4.6M | 801.3K | ⭐⭐⭐ Medium |
| Automotive | 20.0M | 8.0M | 2.0M | ⭐⭐⭐⭐ Hard |

## 🚀 Quick Start with All_Beauty

1. **Download the files:**
   - Go to: https://amazon-reviews-2023.github.io/
   - Find "All_Beauty" in the category table
   - Download both:
     - `review_All_Beauty.jsonl.gz`
     - `meta_All_Beauty.jsonl.gz`

2. **Place in data directory:**
   ```
   data/
   ├── review_All_Beauty.jsonl.gz
   └── meta_All_Beauty.jsonl.gz
   ```

3. **Update main.py** (or use the category parameter):
   ```python
   CATEGORY = "All_Beauty"
   ```

4. **Run:**
   ```bash
   python src/main.py
   ```

## 💡 Tips

- **Start with All_Beauty** - It's the smallest and fastest to process
- **Limit reviews initially** - Use `max_reviews=10000` for quick testing
- **Gradually increase** - Once it works, process more reviews
- **Memory considerations** - Larger categories need more RAM

## 📥 Download Links Format

For any category, the files follow this pattern:
- Review: `review_{Category_Name}.jsonl.gz`
- Metadata: `meta_{Category_Name}.jsonl.gz`

Example for All_Beauty:
- `review_All_Beauty.jsonl.gz`
- `meta_All_Beauty.jsonl.gz`

Note: Category names use underscores (e.g., `Arts_Crafts_and_Sewing`)

