use std::fs;
use std::io::{self, Error, ErrorKind, Read};
use std::path::Path;
use std::fmt::{self, Display, Formatter};
use std::str::FromStr;

const BRANCH_REFS_DIRECTORY: &str = ".git/refs/heads";

fn get_branch_head(branch: &str) -> io::Result<String> {
  let ref_file = Path::new(BRANCH_REFS_DIRECTORY).join(branch);
  fs::read_to_string(ref_file)
}

const HEAD_FILE: &str = ".git/HEAD";

const REF_PREFIX: &str = "ref: refs/heads/";

fn get_head() -> io::Result<Head> {
  use Head::*;

  let hash_contents = fs::read_to_string(HEAD_FILE)?;
  // Remove trailing newline
  let hash_contents = hash_contents.trim_end();
  // If .git/HEAD starts with `ref: refs/heads/`, it's a branch name.
  // Otherwise, it should be a commit hash.
  Ok(match hash_contents.strip_prefix(REF_PREFIX) {
    Some(branch) => Branch(branch.to_string()),
    _ => {
      let hash = Hash::from_str(hash_contents)?;
      Commit(hash)
    }
  })
}

const HASH_BYTES: usize = 20;

// A (commit) hash is a 20-byte identifier.
// We will see that git also gives hashes to other things.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Hash([u8; HASH_BYTES]);

// The head is either at a specific commit or a named branch
enum Head {
  Commit(Hash),
  Branch(String),
}

impl Head {
  fn get_hash(&self) -> io::Result<Hash> {
    use Head::*;

    match self {
      Commit(hash) => Ok(*hash),
      Branch(branch) => {
        // Copied from get_branch_head()
        let ref_file = Path::new(BRANCH_REFS_DIRECTORY).join(branch);
        let hash_contents = fs::read_to_string(ref_file)?;
        Hash::from_str(hash_contents.trim_end())
      }
    }
  }
}

impl FromStr for Hash {
  type Err = Error;

  fn from_str(hex_hash: &str) -> io::Result<Self> {
    // Parse a hexadecimal string like "af64eba00e3cfccc058403c4a110bb49b938af2f"
    // into  [0xaf, 0x64, ..., 0x2f]. Returns an error if the string is invalid.

    // ...
    if hex_hash.len() != HASH_BYTES * 2 {
      return Err(Error::new(ErrorKind::Other, format!("malformed hex hash {}", hex_hash)));
    }
    let hex_hash = hex_hash.as_bytes();
    let mut res = [0; HASH_BYTES];
    for i in 0..HASH_BYTES {
      let b1 = hex_char_value(hex_hash[2*i as usize]).expect("error");
      let b2 = hex_char_value(hex_hash[2*i+1 as usize]).expect("error 2");
      res[i] = b1 << 4 | b2;
    }

    Ok(Hash(res))
  }
}

fn hex_char_value(hex_char: u8) -> Option<u8> {
  match hex_char {
    b'0'..=b'9' => Some(hex_char - b'0'),
    b'a'..=b'f' => Some(hex_char - b'a' + 10),
    _ => None
  }
}

impl Display for Hash {
  fn fmt(&self, f: &mut Formatter) -> fmt::Result {
    // Turn the hash back into a hexadecimal string
    for byte in self.0 {
      write!(f, "{:02x}", byte)?;
    }
    Ok(())
  }
}

#[test]
fn test_hash_empty() {
  let hash = Hash::from_str("");
  assert_eq!(hash.is_err(), true);
}

#[test]
fn test_hash_valid() {
  let hash = Hash::from_str("af64eba00e3cfccc058403c4a110bb49b938af2f");
  assert_eq!(hash.is_ok(), true);
}

fn main() -> io::Result<()> {
  let head = get_head()?;
  let head_hash = head.get_hash()?;
  println!("Head hash: {}", head_hash);
  Ok(())
}

