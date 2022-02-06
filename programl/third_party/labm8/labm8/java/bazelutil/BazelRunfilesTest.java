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

import static org.junit.Assert.assertEquals;

import org.junit.Test;

public class BazelRunfilesTest {

  @Test
  public void getDataPathFileExists() throws Exception {
    BazelRunfiles.getDataPath("phd/labm8/java/test_data/hello_world.txt");
  }

  @Test(expected = IllegalArgumentException.class)
  public void getDataPathFileDoesNotExist() throws Exception {
    BazelRunfiles.getDataPath("this/file/does/not/exist");
  }

  @Test
  public void getDataStringWithFile() throws Exception {
    final String text = BazelRunfiles.getDataString("phd/labm8/java/test_data/hello_world.txt");
    assertEquals(text, "Hello, world!\n");
  }

  @Test
  public void getDataStringOrDieWithFile() throws Exception {
    final String text =
        BazelRunfiles.getDataStringOrDie("phd/labm8/java/test_data/hello_world.txt");
    assertEquals(text, "Hello, world!\n");
  }
}
