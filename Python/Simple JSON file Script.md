### 1. **`.lower()`**

- Converts all characters in the string to lowercase.
- Example: `"Hello, World!"` → `"hello, world!"`

---

### 2. **`.translate(str.maketrans("", "", string.punctuation))`**

#### a. `str.maketrans("", "", string.punctuation)`

- This creates a **translation table** for the `translate()` method.
- The first two arguments (`""`, `""`) are empty, meaning no character-to-character mapping.
- The **third argument** (`string.punctuation`) is a string of all punctuation characters (like `!"#$%&'()*+,-./:;<=>?@[\]^_{|}~`).
- This tells Python: **delete all characters in `string.punctuation`** from the string.

#### b. `.translate(...)`

- Applies the translation table to the string.
- **Removes all punctuation** characters.
- Example:  
    `"hello, world!"` → `"hello world"`

---

### 3. **`.split()`**

- Splits the string into a list of words (tokens) using whitespace.
- Example:  
    `"hello world"` → `["hello", "world"]`

---

## **Full Example**

Suppose `text = "How are you?"`

- `text.lower()` → `"how are you?"`
- `.translate(...)` → `"how are you"` (removes `?`)
- `.split()` → `["how", "are", "you"]`