import re
import os

def list_files(path=os.getcwd(), pattern=None, all_files=False, full_names=False, recursive=False, ignore_case=False, include_dirs=False, no_dot=False):
  """
  This list_files() function has the following arguments:
  
  path: the directory to search. The default value is '.', which means the current directory.
  pattern: a wildcard pattern to filter the results. The default value is None, which means no pattern is used.
  all_files: a boolean value that specifies whether to include directories in the results. The default value is False, which means directories are excluded.
  full_names: a boolean value that specifies whether to return the full file paths or just the file names. The default
  """

  if pattern is None:
    pattern = '.*'
  else:
    pattern = pattern.replace('.', '\\.').replace('*', '.*')

  regex = re.compile(pattern, re.IGNORECASE if ignore_case else 0)

  matches = []
  if recursive:
    for root, dirnames, filenames in os.walk(path):
      if no_dot and root.startswith('.'):
        continue
      for filename in filenames:
        if regex.search(filename):
          if full_names:
            matches.append(os.path.join(root, filename))
          else:
            matches.append(filename)
      if include_dirs:
        for dirname in dirnames:
          if regex.search(dirname):
            if full_names:
              matches.append(os.path.join(root, dirname))
            else:
              matches.append(dirname)
  else:
    filenames = os.listdir(path)
    if no_dot:
      filenames = [f for f in filenames if not f.startswith('.')]
    for filename in filenames:
      if regex.search(filename):
        if full_names:
          matches.append(os.path.join(path, filename))
        else:
          matches.append(filename)
    if include_dirs:
      dirnames = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
      if no_dot:
        dirnames = [d for d in dirnames if not d.startswith('.')]
      for dirname in dirnames:
        if regex.search(dirname):
          if full_names:
            matches.append(os.path.join(path, dirname))
          else:
            matches.append(dirname)

  if not all_files:
    matches = [m for m in matches if not os.path.isdir(m)]

  return matches


import re
import os

def list_files_fast(path=os.getcwd(), pattern=None, all_files=False, full_names=False, recursive=False, ignore_case=False, include_dirs=False, no_dot=False):
  if pattern is None:
    pattern = '.*'
  else:
    pattern = pattern.replace('.', '\\.').replace('*', '.*')

  regex = re.compile(pattern, re.IGNORECASE if ignore_case else 0)

  def scan(dir):
    with os.scandir(dir) as it:
      for entry in it:
        if entry.name.startswith('.') and no_dot:
          continue
        if entry.is_file() or (all_files and entry.is_dir()):
          if regex.search(entry.name):
            if full_names:
              yield entry.path
            else:
              yield entry.name
        elif entry.is_dir() and recursive:
          yield from scan(entry.path)

  return list(scan(path))



import timeit
def test_list_files(func, pattern):
  a=timeit.timeit(lambda: func(pattern=pattern, recursive=True), number=100)
  b=timeit.timeit(lambda: func(pattern=pattern, recursive=False), number=100)
  c=timeit.timeit(lambda: func(pattern=pattern, recursive=True, ignore_case=True), number=100)
  d=timeit.timeit(lambda: func(pattern=pattern, recursive=False, ignore_case=True), number=100)
  e=timeit.timeit(lambda: func(pattern=pattern, recursive=True, full_names=True), number=100)
  f=timeit.timeit(lambda: func(pattern=pattern, recursive=False, full_names=True), number=100)
  return [a, b, c, d, e, f]
# the differences are negligible 
fast = test_list_files(list_files_fast, pattern="l$")
slow = test_list_files(list_files, pattern="l$")
