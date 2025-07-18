module {
  func.func @main(%arg0: tensor<67x74x80x16x77x78xf32>, %arg1: tensor<67x74x80x42x77x78xf32>, %arg2: tensor<25x29x3x67xf32>) -> (tensor<25x29x3xi32>, tensor<67x74x80x58x77x78xf32>) {
    %0 = tosa.concat %arg0, %arg1 {axis = 3 : i32} : (tensor<67x74x80x16x77x78xf32>, tensor<67x74x80x42x77x78xf32>) -> tensor<67x74x80x58x77x78xf32>
    %1 = tosa.argmax %arg2 {axis = 3 : i32} : (tensor<25x29x3x67xf32>) -> tensor<25x29x3xi32>
    %2 = tosa.log %0 : (tensor<67x74x80x58x77x78xf32>) -> tensor<67x74x80x58x77x78xf32>
    %3 = tosa.exp %2 : (tensor<67x74x80x58x77x78xf32>) -> tensor<67x74x80x58x77x78xf32>
    %4 = tosa.exp %3 : (tensor<67x74x80x58x77x78xf32>) -> tensor<67x74x80x58x77x78xf32>
    return %1, %4 : tensor<25x29x3xi32>, tensor<67x74x80x58x77x78xf32>
  }
}
