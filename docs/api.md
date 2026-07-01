# API

Primary entry point:

```python
from emma_3dgenome import EmmaRestorer
```

Core methods:

- `restore(matrix, mask, regions=None)`
- `restore_from_file(path, ...)`
- `restore_auto(matrix, ...)`
- `reconstruct(matrix, mode="conservative", blend=0.2)`
