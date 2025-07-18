module {
  func.func @main(%arg0: tensor<38x84x21x25x17xf32>, %arg1: tensor<13x21x35x60x69xi64>, %arg2: tensor<1x21x35x1x1xi64>, %arg3: tensor<41x48x84xi1>) -> (tensor<41x48x84xi1>, tensor<13x21x35x60x69xi64>, tensor<38x84x21x25x17xf32>) {
    %0 = tosa.exp %arg0 : (tensor<38x84x21x25x17xf32>) -> tensor<38x84x21x25x17xf32>
    %1 = tosa.logical_left_shift %arg1, %arg2 : (tensor<13x21x35x60x69xi64>, tensor<1x21x35x1x1xi64>) -> tensor<13x21x35x60x69xi64>
    %2 = tosa.tanh %0 : (tensor<38x84x21x25x17xf32>) -> tensor<38x84x21x25x17xf32>
    %3 = tosa.minimum %1, %1 : (tensor<13x21x35x60x69xi64>, tensor<13x21x35x60x69xi64>) -> tensor<13x21x35x60x69xi64>
    %4 = tosa.rsqrt %2 : (tensor<38x84x21x25x17xf32>) -> tensor<38x84x21x25x17xf32>
    %5 = tosa.logical_not %arg3 : (tensor<41x48x84xi1>) -> tensor<41x48x84xi1>
    %6 = tosa.minimum %3, %1 : (tensor<13x21x35x60x69xi64>, tensor<13x21x35x60x69xi64>) -> tensor<13x21x35x60x69xi64>
    %7 = tosa.tanh %4 : (tensor<38x84x21x25x17xf32>) -> tensor<38x84x21x25x17xf32>
    return %5, %6, %7 : tensor<41x48x84xi1>, tensor<13x21x35x60x69xi64>, tensor<38x84x21x25x17xf32>
  }
}
