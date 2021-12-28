use std::env;
use std::io::{self, Error, ErrorKind, Read, Seek, SeekFrom};
use std::path::Path;
use std::fmt::{self, Display, Formatter};
use std::fs::{self, File};
use std::str::{self, FromStr};
use std::ffi::OsStr;
use std::convert::TryFrom;
use flate2::read::ZlibDecoder;
use sha1::{Sha1, Digest};

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

const PACKS_DIRECTORY: &str = ".git/objects/pack";

// Reads a fixed number of bytes from a stream.
// Rust's "const generics" make this function very useful.
fn read_bytes<R: Read + ?Sized, const N: usize>(stream: &mut R)
  -> io::Result<[u8; N]>
{
  let mut bytes = [0; N];
  stream.read_exact(&mut bytes)?;
  Ok(bytes)
}

// Reads a big-endian 32-bit (4-byte) integer from a stream
fn read_u32<R: Read>(stream: &mut R) -> io::Result<u32> {
  let bytes = read_bytes(stream)?;
  Ok(u32::from_be_bytes(bytes))
}

fn read_u64<R: Read>(stream: &mut R) -> io::Result<u64> {
  let bytes = read_bytes(stream)?;
  Ok(u64::from_be_bytes(bytes))
}

// Read an object hash from a stream
fn read_hash<R: Read>(stream: &mut R) -> io::Result<Hash> {
  let bytes = read_bytes(stream)?;
  Ok(Hash(bytes))
}

fn read_pack_index(file: &str) -> io::Result<()> {
  let file_path = Path::new(PACKS_DIRECTORY).join(file);
  let mut file = File::open(file_path)?;

  // Check index header
  let magic = read_bytes::<dyn Read, 4>(&mut file)?;
  //assert_eq!(magic, *b"\xfftOc");
  let version = read_u32(&mut file)?;
  assert_eq!(version, 2);
  
  // For each of the 256 possible first bytes `b` of a hash,
  // read the cumulative number of objects with first byte <= `b`
  let mut cumulative_objects = [0; 1 << u8::BITS];
  for objects in &mut cumulative_objects {
    *objects = read_u32(&mut file)?;
  }

  // Read the hash of each of the objects.
  // Check that the hashes have the correct first byte and are sorted.
  let mut previous_objects = 0;
  for (first_byte, &objects) in cumulative_objects.iter().enumerate() {
    println!("{:#02x?} -> {:?}", first_byte, &objects);
    // The difference in the cumulative number of objects
    // is the number of objects with this first byte
    let mut previous_hash = None;
    for _ in 0..(objects - previous_objects) {
      // We already know the first byte of the hash, so ensure it matches
      let hash = read_hash(&mut file)?;
      assert_eq!(hash.0[0], first_byte as u8);
      if let Some(previous_hash) = previous_hash {
        assert!(hash > previous_hash);
      }
      previous_hash = Some(hash);
    }
    previous_objects = objects;
  }
  //`cumulative_objects[255]` is the total number of objects
  let total_objects = previous_objects;
  println!("total_objects {}", total_objects);

  // Read a checksum of the packed data for each object
  for _ in 0..total_objects {
    let _crc32 = read_u32(&mut file)?;
  }

  // Read the offset of each object within the packfile
  for _ in 0..total_objects {
    let _pack_offset = read_u32(&mut file)?;
    // TODO: there's one more step needed to read large pack offsets
  }
  Ok(())
}

const fn cumulative_objects_position(first_byte: u8) -> u64 {
  // Skip the magic bytes, version number,
  // and previous cumulative object counts
  4 + 4 + first_byte as u64 * 4
}

fn seek(file: &mut File, offset: u64) -> io::Result<()> {
  file.seek(SeekFrom::Start(offset))?;
  Ok(())
}

// Gets lower and upper bounds on the index of an object hash
// in an index file, using the cumulative object counts
fn get_object_index_bounds(index_file: &mut File, hash: Hash)
  -> io::Result<(u32, u32)>
{
  // The previous cumulative object count is the lower bound (inclusive)
  let first_byte = hash.0[0];
  let index_lower_bound = if first_byte == 0 {
    seek(index_file, cumulative_objects_position(0))?;
    // There aren't any hashes with a lower first byte than 0
    0
  }
  else {
    seek(index_file, cumulative_objects_position(first_byte - 1))?;
    read_u32(index_file)?
  };
  // The next cumulative object count is the upper bound (exclusive)
  let index_upper_bound = read_u32(index_file)?;
  Ok((index_lower_bound, index_upper_bound))
}

const TOTAL_OBJECTS_POSITION: u64 = cumulative_objects_position(u8::MAX);
fn hash_position(object_index: u32) -> u64 {
  // Skip the cumulative object counts and the previous hashes
  // +4 to skip the last object bucket's count
  TOTAL_OBJECTS_POSITION + 4 + object_index as u64 * HASH_BYTES as u64
}

fn get_object_index(index_file: &mut File, hash: Hash) -> io::Result<Option<u32>>
{
  use std::cmp::Ordering::*;

  // Track the range of possible indices for the object hash.
  // (left_index is inclusive, right_index is exclusive)
  let (mut left_index, mut right_index) = get_object_index_bounds(index_file, hash)?;
  while left_index < right_index {
    // Compare with the object hash in the middle of the range
    let mid_index = left_index + (right_index - left_index) / 2;
    seek(index_file, hash_position(mid_index))?;
    let mid_hash = read_hash(index_file)?;
    match hash.cmp(&mid_hash) {
      Less => right_index = mid_index, // the object is to the left
      Equal => return Ok(Some(mid_index)), // we found the object
      Greater => left_index = mid_index + 1, // the object is to the right
    }
  }
  // If the range is empty, the object isn't in the index file
  Ok(None)
}

fn crc32_position(total_objects: u32, object_index: u32) -> u64 {
  // Skip the hashes and previous CRC-32s
  hash_position(total_objects) + object_index as u64 * 4
}
fn offset_position(total_objects: u32, object_index: u32) -> u64 {
  // Skip the CRC-32s and previous object offsets
  crc32_position(total_objects, total_objects) + object_index as u64 * 4
}
fn long_offset_position(total_objects: u32, offset_index: u32) -> u64 {
  // Skip the short object offsets and previous long object offsets
  offset_position(total_objects, total_objects) + offset_index as u64 * 8
}

// The most significant bit of a 32-bit integer
const LONG_OFFSET_FLAG: u32 = 1 << 31;

fn get_pack_offset_at_index(index_file: &mut File, object_index: u32)
  -> io::Result<u64>
{
  seek(index_file, TOTAL_OBJECTS_POSITION)?;
  let total_objects = read_u32(index_file)?;
  seek(index_file, offset_position(total_objects, object_index))?;
  let pack_offset = read_u32(index_file)?;
  if pack_offset & LONG_OFFSET_FLAG == 0 {
    // If the flag bit isn't set, the offset is just a 32-bit offset
    Ok(pack_offset as u64)
  }
  else {
    // If the flag bit is set, the rest of the offset
    // is an index into the 64-bit offsets
    let offset_index = pack_offset & !LONG_OFFSET_FLAG;
    seek(index_file, long_offset_position(total_objects, offset_index))?;
    read_u64(index_file)
  }
}

const INDEX_FILE_SUFFIX: &str = ".idx";

// Gets the offset of an object in a packfile.
// `pack` is the name of the packfile, without ".idx" or ".pack".
fn get_pack_offset(pack_dir: &str, pack: &str, hash: Hash) -> io::Result<Option<u64>> {
  let path = Path::new(pack_dir).join(pack.to_string() + INDEX_FILE_SUFFIX);
  println!("Path: {:?}", path);
  let mut file = File::open(path)?;
  let object_index = get_object_index(&mut file, hash)?;
  let object_index = match object_index {
    Some(object_index) => object_index,
    _ => return Ok(None),
  };

  let pack_offset = get_pack_offset_at_index(&mut file, object_index)?;
  Ok(Some(pack_offset))
}

// Each byte contributes 7 bits of data
const VARINT_ENCODING_BITS: u8 = 7;
// The upper bit indicates whether there are more bytes
const VARINT_CONTINUE_FLAG: u8 = 1 << VARINT_ENCODING_BITS;

// Read 7 bits of data and a flag indicating whether there are more
fn read_varint_byte<R: Read>(stream: &mut R) -> io::Result<(u8, bool)> {
  let [byte] = read_bytes(stream)?;
  let value = byte & !VARINT_CONTINUE_FLAG;
  let more_bytes = byte & VARINT_CONTINUE_FLAG != 0;
  Ok((value, more_bytes))
}

// Read a "size encoding" variable-length integer.
// (There's another slightly different variable-length format
// called the "offset encoding".)
fn read_size_encoding<R: Read>(stream: &mut R) -> io::Result<usize> {
  let mut value = 0;
  let mut length = 0; // the number of bits of data read so far
  loop {
    let (byte_value, more_bytes) = read_varint_byte(stream)?;
    // Add in the data bits
    value |= (byte_value as usize) << length;
    // Stop if this is the last byte
    if !more_bytes {
      return Ok(value)
    }

    length += VARINT_ENCODING_BITS;
  }
}

const PACK_FILE_SUFFIX: &str = ".pack";

#[derive(Clone, Copy, Debug)]
enum ObjectType {
  Commit,
  Tree,
  Blob,
  Tag,
}

// A packed object can either store the object's contents directly,
// or as a "delta" from another object
enum PackObjectType {
  Base(ObjectType),
  OffsetDelta,
  HashDelta,
}

// An object, which may be read from a packfile or an unpacked file
#[derive(Debug)]
struct Object {
  object_type: ObjectType,
  contents: Vec<u8>,
}

const COMMIT_OBJECT_TYPE: &[u8] = b"commit";
const TREE_OBJECT_TYPE: &[u8] = b"tree";
const BLOB_OBJECT_TYPE: &[u8] = b"blob";
const TAG_OBJECT_TYPE: &[u8] = b"tag";

impl Object {
  // Compute the hash that an object would have, given its type and contents
  fn hash(&self) -> Hash {
    use ObjectType::*;

    let mut hasher = Sha1::new();
    hasher.update(match self.object_type {
        Commit => COMMIT_OBJECT_TYPE,
        Tree => TREE_OBJECT_TYPE,
        Blob => BLOB_OBJECT_TYPE,
        Tag => TAG_OBJECT_TYPE,
      });
    hasher.update(b" ");
    hasher.update(self.contents.len().to_string());
    hasher.update(b"\0");
    hasher.update(&self.contents);
    let hash = hasher.finalize();

    Hash(<[u8; HASH_BYTES]>::try_from(hash.as_slice()).unwrap())
  }
}

// Remove the `.idx` suffix from an index filename.
// Returns None if not an index file.
fn strip_index_file_name(file_name: &OsStr) -> Option<&str> {
  let file_name = file_name.to_str()?;
  file_name.strip_suffix(INDEX_FILE_SUFFIX)
}

// Read a packed object from the packs directory
fn read_packed_object(hash: Hash, pack_dir: &str) -> io::Result<Object> {
  // Try every file in the packs directory
  for pack_or_index in fs::read_dir(pack_dir)? {
    let pack_or_index = pack_or_index?;
    let file_name = pack_or_index.file_name();
    // Skip any non-index files
    let pack = match strip_index_file_name(&file_name) {
      Some(pack) => pack,
      _ => continue,
    };

    // Skip the pack if the object is not in the index
    let pack_offset = get_pack_offset(pack_dir, pack, hash)?;
    let pack_offset = match pack_offset {
      Some(pack_offset) => pack_offset,
      _ => continue,
    };

    // If the object is found in the index, read it from the pack
    return unpack_object(pack_dir, pack, pack_offset)
  }
  Err(make_error(&format!("Object {} not found", hash)))
}

// Read an unpacked object from the objects directory
fn read_unpacked_object(hash: Hash) -> io::Result<Object> {
  // Modified from read_object() and check_header() in the last post
  // ...
  Err(Error::new(ErrorKind::NotFound, "not found"))
}

// Shorthand for making an io::Error
fn make_error(message: &str) -> Error {
  Error::new(ErrorKind::Other, message)
}

// Read an object when we don't know if it's packed or unpacked
// from the pack_dir
fn generic_read_object(hash: Hash, pack_dir: &str) -> io::Result<Object> {
  let object = match read_unpacked_object(hash) {
    Ok(object) => object,
    Err(err) if err.kind() == ErrorKind::NotFound => {
      read_packed_object(hash, pack_dir)?
    }
    err => return err,
  };

  // Verify that the object has the SHA-1 hash we expected
  let object_hash = object.hash();
  if object_hash != hash {
    return Err(make_error(
      &format!("Object {} has wrong hash {}", hash, object_hash)
    ))
  }

  Ok(object)
}

fn read_pack_object(pack_file: &mut File, offset: u64, pack_dir: &str) -> io::Result<Object> {
  use ObjectType::*;
  use PackObjectType::*;
  seek(pack_file, offset)?;
  
  let (object_type, size) = read_type_and_size(pack_file)?;
  let object_type = match object_type {
    1 => Base(Commit),
    2 => Base(Tree),
    3 => Base(Blob),
    4 => Base(Tag),
    6 => OffsetDelta,
    7 => HashDelta,
    _ => {
      return Err(make_error(&format!("Invalid object type: {}", object_type)))
    }
  };
  match object_type {
    Base(object_type) => {
      // The object contents are zlib-compressed
      let mut contents = Vec::with_capacity(size);
      ZlibDecoder::new(pack_file).read_to_end(&mut contents)?;
      if contents.len() != size {
        return Err(make_error("Incorrect object size"))
      }

      Ok(Object { object_type, contents })
    }
    OffsetDelta => {
      let delta_offset = read_offset_encoding(pack_file)?;
      let base_offset = offset.checked_sub(delta_offset).ok_or_else(|| {
        make_error("Invalid OffsetDelta offset")
      })?;
      // Save and restore the offset since read_pack_offset() will change it
      let offset = get_offset(pack_file)?;
      let base_object = read_pack_object(pack_file, base_offset, pack_dir)?;
      seek(pack_file, offset)?;
      apply_delta(pack_file, &base_object)
    }
    HashDelta => {
      let hash = read_hash(pack_file)?;
      let base_object = generic_read_object(hash, pack_dir)?; // to implement shortly
      apply_delta(pack_file, &base_object)
    }
  }
}

fn get_offset(file: &mut File) -> io::Result<u64> {
  file.seek(SeekFrom::Start(0))
}

fn unpack_object(pack_dir: &str, pack: &str, offset: u64) -> io::Result<Object> {
  let path = Path::new(pack_dir).join(pack.to_string() + PACK_FILE_SUFFIX);
  let mut file = File::open(&path)?;
  let pack_dir = path.parent().ok_or_else(|| make_error("failed to get parent"))?;
  let pack_dir = pack_dir.to_str().ok_or_else(|| make_error("failed to get path str"))?;
  read_pack_object(&mut file, offset, pack_dir)
}


// The number of bits storing the object type
const TYPE_BITS: u8 = 3;
// The number of bits of the object size in the first byte.
// Each additional byte has VARINT_ENCODING_BITS of size.
const TYPE_BYTE_SIZE_BITS: u8 = VARINT_ENCODING_BITS - TYPE_BITS;

// Read the lower `bits` bits of `value`
fn keep_bits(value: usize, bits: u8) -> usize {
  value & ((1 << bits) - 1)
}

fn read_type_and_size<R: Read>(stream: &mut R) -> io::Result<(u8, usize)> {
  // Object type and uncompressed pack data size
  // are stored in a "size-encoding" variable-length integer.
  // Bits 4 through 6 store the type and the remaining bits store the size.
  let value = read_size_encoding(stream)?;
  let object_type = keep_bits(value >> TYPE_BYTE_SIZE_BITS, TYPE_BITS) as u8;
  let size = keep_bits(value, TYPE_BYTE_SIZE_BITS)
           | (value >> VARINT_ENCODING_BITS << TYPE_BYTE_SIZE_BITS);
  Ok((object_type, size))
}

fn read_offset_encoding<R: Read>(stream: &mut R) -> io::Result<u64> {
  let mut value = 0;
  loop {
    let (byte_value, more_bytes) = read_varint_byte(stream)?;
    // Add the new bits at the *least* significant end of the value
    value = (value << VARINT_ENCODING_BITS) | byte_value as u64;
    if !more_bytes {
      return Ok(value)
    }

    // Increase the value if there are more bytes, to avoid redundant encodings
    value += 1;
  }
}

const COPY_INSTRUCTION_FLAG: u8 = 1 << 7;
const COPY_OFFSET_BYTES: u8 = 4;
const COPY_SIZE_BYTES: u8 = 3;
const COPY_ZERO_SIZE: usize = 0x10000;

// Read an integer of up to `bytes` bytes.
// `present_bytes` indicates which bytes are provided. The others are 0.
fn read_partial_int<R: Read>(
  stream: &mut R, bytes: u8, present_bytes: &mut u8
) -> io::Result<usize> {
  let mut value = 0;
  for byte_index in 0..bytes {
    // Use one bit of `present_bytes` to determine if the byte exists
    if *present_bytes & 1 != 0 {
      let [byte] = read_bytes(stream)?;
      value |= (byte as usize) << (byte_index * 8);
    }
    *present_bytes >>= 1;
  }
  Ok(value)
}

// Reads a single delta instruction from a stream
// and appends the relevant bytes to `result`.
// Returns whether the delta stream still had instructions.
fn apply_delta_instruction<R: Read>(
  stream: &mut R, base: &[u8], result: &mut Vec<u8>
) -> io::Result<bool> {
  // Check if the stream has ended, meaning the new object is done
  let instruction = match read_bytes(stream) {
    Ok([instruction]) => instruction,
    Err(err) if err.kind() == ErrorKind::UnexpectedEof => return Ok(false),
    Err(err) => return Err(err),
  };
  if instruction & COPY_INSTRUCTION_FLAG == 0 {
    // Data instruction; the instruction byte specifies the number of data bytes
    if instruction == 0 {
      // Appending 0 bytes doesn't make sense, so git disallows it
      return Err(make_error("Invalid data instruction"))
    }

    // Append the provided bytes
    let mut data = vec![0; instruction as usize];
    stream.read_exact(&mut data)?;
    result.extend_from_slice(&data);
  }
  else {
    // Copy instruction
    let mut nonzero_bytes = instruction;
    let offset =
      read_partial_int(stream, COPY_OFFSET_BYTES, &mut nonzero_bytes)?;
    let mut size =
      read_partial_int(stream, COPY_SIZE_BYTES, &mut nonzero_bytes)?;
    if size == 0 {
      // Copying 0 bytes doesn't make sense, so git assumes a different size
      size = COPY_ZERO_SIZE;
    }
    // Copy bytes from the base object
    let base_data = base.get(offset..(offset + size)).ok_or_else(|| {
      make_error("Invalid copy instruction")
    })?;
    result.extend_from_slice(base_data);
  }
  Ok(true)
}

fn apply_delta(pack_file: &mut File, base: &Object) -> io::Result<Object> {
  let Object { object_type, contents: ref base } = *base;
  let mut delta = ZlibDecoder::new(pack_file);
  let base_size = read_size_encoding(&mut delta)?;
  if base.len() != base_size {
    return Err(make_error("Incorrect base object length"))
  }

  let result_size = read_size_encoding(&mut delta)?;
  let mut result = Vec::with_capacity(result_size);
  while apply_delta_instruction(&mut delta, base, &mut result)? {}
  if result.len() != result_size {
    return Err(make_error("Incorrect object length"))
  }

  // The object type is the same as the base object
  Ok(Object { object_type, contents: result })
}

fn main() -> io::Result<()> {
  //let args: Vec<_> = env::args().collect();
  //let [_, pack, object] = <[String; 3]>::try_from(args).unwrap();
  //let offset = get_pack_offset(&pack, Hash::from_str(&object).unwrap())?;
  //let Object { object_type, contents } = unpack_object(&pack, offset.unwrap())?;
  //println!("Type: {:?}", object_type);
  //println!("{}", String::from_utf8_lossy(&contents));
  let args: Vec<_> = env::args().collect();
  let [_, pack, hash] = <[String; 3]>::try_from(args).unwrap();
  let hash = Hash::from_str(&hash)?;
  let Object { object_type, contents } = generic_read_object(hash, &pack)?;
  println!("Object type: {:?}", object_type);
  println!("{}", String::from_utf8_lossy(&contents));
  Ok(())
}

