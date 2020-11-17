int Fib(int x) {
  switch (x) {
    case 0:
      return 0;
    case 1:
      return 1;
    default:
      return Fib(x - 1) + Fib(x - 2);
  }
}
