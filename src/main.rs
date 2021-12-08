use std::fs;
use std::io;
use std::path::Path;

const BRANCH_REFS_DIRECTORY: &str = ".git/refs/heads";

fn get_branch_head(branch: &str) -> io::Result<String> {
  let ref_file = Path::new(BRANCH_REFS_DIRECTORY).join(branch);
  fs::read_to_string(ref_file)
}

const HEAD_FILE: &str = ".git/HEAD";

fn get_head() -> io::Result<String> {
  fs::read_to_string(HEAD_FILE)
}

fn main() -> io::Result<()> {
  let main_head = get_branch_head("master")?;
  println!("main: {:?}", main_head);
  Ok(())
}
