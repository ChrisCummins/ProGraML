"""A "hello world" app."""
from labm8.py import app

FLAGS = app.FLAGS

app.DEFINE_boolean("create_file", False, "Create /tmp/hello.txt file.")
app.DEFINE_string("hello_to", None, "Say hello to someone.")


def main():
    """Main entry point."""
    if FLAGS.create_file:
        with open("/tmp/hello.txt", "w") as f:
            f.write("Hello, world!\n")
    elif FLAGS.hello_to:
        print(f"Hello to {FLAGS.hello_to}!")
    else:
        print("Hello, world!")


if __name__ == "__main__":
    app.Run(main)
