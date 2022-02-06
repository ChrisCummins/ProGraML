// Copyright 2014-2020 Chris Cummins <chrisc.101@gmail.com>.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package labm8.java.bazelutil;

import com.google.devtools.build.runfiles.Runfiles;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

public class BazelRunfiles {

  public static String getDataPath(Runfiles runfiles, final String path) throws IOException {
    final String runfiles_path = runfiles.rlocation(path);
    final File file = new File(runfiles_path);
    if (!file.exists()) {
      throw new IllegalArgumentException("Runfile not found: `" + path + "`");
    }
    return runfiles_path;
  }

  public static String getDataPath(final String path) throws IOException {
    return getDataPath(createRunfiles(), path);
  }

  public static String getDataPathOrDie(final String path) {
    try {
      return getDataPath(createRunfilesOrDie(), path);
    } catch (IOException e) {
      System.err.println("Failed to resolve runfiles path: `" + path + "`");
      System.exit(1);
      return null;
    }
  }

  public static String getDataString(final String path) throws IOException {
    return readFileToString(getDataPath(path));
  }

  public static String getDataStringOrDie(final String path) {
    try {
      return readFileToString(getDataPathOrDie(path));
    } catch (IOException e) {
      System.err.println("Failed to read runfile: `" + path + "`");
      System.exit(1);
      return null;
    }
  }

  private static Runfiles createRunfiles() throws IOException {
    return Runfiles.create();
  }

  private static Runfiles createRunfilesOrDie() {
    try {
      return createRunfiles();
    } catch (IOException e) {
      System.err.println("Failed to create runfiles instance!");
      System.exit(1);
      return null;
    }
  }

  private static String readFileToString(final String path) throws IOException {
    InputStream is = new FileInputStream(path);
    BufferedReader buf = new BufferedReader(new InputStreamReader(is));

    String line = buf.readLine();
    StringBuilder sb = new StringBuilder();
    while (line != null) {
      sb.append(line).append("\n");
      line = buf.readLine();
    }

    return sb.toString();
  }
}
