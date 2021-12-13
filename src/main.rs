use std::fs;
use std::io::{self, Error, ErrorKind, Read};
use std::path::Path;
use std::fmt::{self, Display, Formatter};
use std::fs::File;
use std::str::FromStr;
use std::str;
use std::convert::TryFrom;
use flate2::read::ZlibDecoder;

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
      let b1 = hex_char_value(hex_hash[2*i as usize]).ok_or_else(|| {
        Error::new(ErrorKind::Other, format!("Malformed hash byte: {}", hex_hash[2*i as usize]))
      })?;
      let b2 = hex_char_value(hex_hash[2*i+1 as usize]).ok_or_else(|| {
        Error::new(ErrorKind::Other, format!("Malformed hash byte: {}", hex_hash[2*i+1 as usize]))
      })?;
    
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

fn hex_to_hash(hex_hash: &[u8]) -> Option<Hash> {
  let mut res = [0; HASH_BYTES];
  for i in 0..HASH_BYTES {
    let b1 = hex_char_value(hex_hash[2*i as usize])?;
    let b2 = hex_char_value(hex_hash[2*i+1 as usize])?;
  
    res[i] = b1 << 4 | b2;
  }

  Some(Hash(res))
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

const OBJECTS_DIRECTORY: &str = ".git/objects";

// Read the byte contents of an object
fn read_object(hash: Hash) -> io::Result<Vec<u8>> {
  // The first 2 characters of the hexadecimal hash form the directory;
  // the rest forms the filename
  let hex_hash = hash.to_string();
  let (directory_name, file_name) = hex_hash.split_at(2);
  let object_file = Path::new(OBJECTS_DIRECTORY)
    .join(directory_name)
    .join(file_name);
  let object_file = File::open(object_file)?;
  let mut contents = vec![];
  ZlibDecoder::new(object_file).read_to_end(&mut contents)?;
  Ok(contents)
}

const COMMIT_HEADER: &[u8] = b"commit ";
const TREE_LINE_PREFIX: &[u8] = b"tree ";
const PARENT_LINE_PREFIX: &[u8] = b"parent ";
const AUTHOR_LINE_PREFIX: &[u8] = b"author ";
const COMMITTER_LINE_PREFIX: &[u8] = b"committer ";

// Some helper functions for parsing objects

fn decimal_char_value(decimal_char: u8) -> Option<u8> {
  match decimal_char {
    b'0'..=b'9' => Some(decimal_char - b'0'),
    _ => None,
  }
}

// Parses a decimal string, e.g. "123", into its value, e.g. 123.
// Returns None if any characters are invalid or the value overflows a usize.
fn parse_decimal(decimal_str: &[u8]) -> Option<usize> {
  let mut value = 0usize;
  for &decimal_char in decimal_str {
    let char_value = decimal_char_value(decimal_char)?;
    value = value.checked_mul(10)?;
    value = value.checked_add(char_value as usize)?;
  }
  Some(value)
}

// Like str::split_once(), split the slice at the next delimiter
fn split_once<T: PartialEq>(slice: &[T], delimiter: T) -> Option<(&[T], &[T])> {
  let index = slice.iter().position(|element| *element == delimiter)?;
  Some((&slice[..index], &slice[index + 1..]))
}

// Checks that an object's header has the expected type, e.g. "commit ",
// and the object size is correct
fn check_header<'a>(object: &'a [u8], header: &[u8]) -> Option<&'a [u8]> {
  let object = object.strip_prefix(header)?;
  let (size, object) = split_once(object, b'\0')?;
  let size = parse_decimal(size)?;
  if object.len() != size {
    return None
  }

  Some(object)
}

#[derive(Debug)]
struct Commit {
  tree: Hash,
  parents: Vec<Hash>,
  author: String, // name, email, and timestamp (not parsed)
  committer: String, // same contents as `author`
  message: String, // includes commit description
}

fn parse_commit(object: &[u8]) -> Option<Commit> {
  let object = check_header(object, COMMIT_HEADER)?;

  let object = object.strip_prefix(TREE_LINE_PREFIX)?;
  let (tree, mut object) = split_once(object, b'\n')?;
  let tree = hex_to_hash(tree)?;

  let mut parents = vec![];
  while let Some(object_rest) = object.strip_prefix(PARENT_LINE_PREFIX) {
    let (parent, object_rest) = split_once(object_rest, b'\n')?;
    let parent = hex_to_hash(parent)?;
    parents.push(parent);
    object = object_rest;
  }

  let object = object.strip_prefix(AUTHOR_LINE_PREFIX)?;
  let (author, object) = split_once(object, b'\n')?;
  let author = String::from_utf8(author.to_vec()).ok()?;

  let object = object.strip_prefix(COMMITTER_LINE_PREFIX)?;
  let (committer, object) = split_once(object, b'\n')?;
  let committer = String::from_utf8(committer.to_vec()).ok()?;

  //println!("commiter");
  //println!("{:?}", str::from_utf8(object).unwrap());
  //let object = object.strip_prefix(b"gpgsig")?;
  //let (sig, object) = split_once(object, b"-----END PGP SIGNATURE-----")?;
  //
  //let object = object.strip_prefix(b"\n")?;
  //let message = String::from_utf8(object.to_vec()).ok()?;
  //println!("commiter done");

  let message = String::from("");
  Some(Commit { tree, parents, author, committer, message })
}

fn read_commit(hash: Hash) -> io::Result<Commit> {
  let object = read_object(hash)?;
  parse_commit(&object).ok_or_else(|| {
    Error::new(ErrorKind::Other, format!("Malformed commit object: {}", hash))
  })
}

const TREE_HEADER: &[u8] = b"tree ";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
  Directory,
  File,
  // We'll ignore other modes for now
}

#[derive(Debug)]
struct TreeEntry {
  mode: Mode,
  name: String,
  hash: Hash,
}

#[derive(Debug)]
struct Tree(Vec<TreeEntry>);

fn parse_tree(object: &[u8]) -> Option<Tree> {
  let mut object = check_header(object, TREE_HEADER)?;
  let mut entries = vec![];
  while !object.is_empty() {
    let (mode, object_rest) = split_once(object, b' ')?;
    let mode = match mode {
      b"40000" => Mode::Directory,
      b"100644" => Mode::File,
      _ => return None,
    };

    let (name, object_rest) = split_once(object_rest, b'\0')?;
    let name = String::from_utf8(name.to_vec()).ok()?;

    let hash = object_rest.get(..HASH_BYTES)?;
    let hash = Hash(*<&[u8; HASH_BYTES]>::try_from(hash).unwrap());
    object = &object_rest[HASH_BYTES..];

    entries.push(TreeEntry { mode, name, hash });
  }
  Some(Tree(entries))
}

fn read_tree(hash: Hash) -> io::Result<Tree> {
  let object = read_object(hash)?;
  parse_tree(&object).ok_or_else(|| {
    Error::new(ErrorKind::Other, format!("Malformed tree object: {}", hash))
  })
}
const BLOB_HEADER: &[u8] = b"blob ";

struct Blob(Vec<u8>);

fn read_blob(hash: Hash) -> io::Result<Blob> {
  let object = read_object(hash)?;
  let bytes = check_header(&object, BLOB_HEADER).ok_or_else(|| {
    Error::new(ErrorKind::Other, format!("Malformed blob object: {}", hash))
  })?;
  Ok(Blob(bytes.to_vec()))
}

fn get_file_blob(tree: Hash, path: &str) -> io::Result<Blob> {
  let mut hash = tree;
  for name in path.split('/') {
    let tree = read_tree(hash)?;
    let entry = tree.0.iter().find(|entry| entry.name == name).ok_or_else(|| {
      Error::new(ErrorKind::Other, format!("No such entry: {}", name))
    })?;
    hash = entry.hash;
  }
  read_blob(hash)
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

  let head_contents = read_object(head_hash)?;

  // Spoiler alert: the commit object is a text file, so print it as a string
  let head_contents = String::from_utf8(head_contents).unwrap();
  println!("Object {} contents:", head_hash);
  println!("{:?}", head_contents);

  println!("read_commit");
  let commit = read_commit(head_hash)?;
  println!("Commit {}:", head_hash);
  println!("{:x?}", commit);

  let blob = get_file_blob(commit.tree, "src/main.rs")?;
  print!("{}", String::from_utf8(blob.0).unwrap()); // assume a text file

  Ok(())
}

