# Error Logs

This document summarizes the errors encountered during the development and execution of the JMMLU Benchmark and their respective solutions.

## 1. JSON Parsing Error
**Error**: 
```
Extra data: line 1 column 24 (char 23)
```

**Cause**: 
This occurred due to an incorrect pattern in the `get_umls_keys()` function.

**Solution**: 
Updated the pattern in `get_umls_keys()` from:
```python
pattern = r"\{(.*?)\}"
```
to:
```python
pattern = r"\*\*({.*?})\*\*"
```
**File**: `umls_rerank_cohere.py`  
**Location**: Line 125

---

## 2. Too Many Requests Error
**Error**:
```
cohere.errors.TooManyRequestsError: status_code: 429, body: data=None message="You are using a Trial key, which is limited to 1000 API calls / month. You can continue to use the Trial key for free or upgrade to a Production key with higher rate limits at 'https://dashboard.cohere.com/api-keys'. Contact us on 'https://discord.gg/XW44jPfYJu' or email us at support@cohere.com with any questions"
```

**Cause**:
This error occurs when the API call limit for the Trial key is exceeded.

---

If further issues arise, please document them here for future debugging and solutions.
