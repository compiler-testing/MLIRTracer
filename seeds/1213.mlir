module {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<42x44x85x24x94xi64>, %arg2: tensor<1x44x1x1x94xi64>, %arg3: tensor<55x8xi1>) -> (tensor<94x24x44x85x42xi64>, tensor<55x8xi1>, tensor<42x44x85x24x94xi64>, tensor<8xi32>, tensor<f32>) {
    %0 = tosa.reciprocal %arg0 : (tensor<f32>) -> tensor<f32>
    %1 = tosa.minimum %arg1, %arg2 : (tensor<42x44x85x24x94xi64>, tensor<1x44x1x1x94xi64>) -> tensor<42x44x85x24x94xi64>
    %2 = "tosa.const"() {values = dense<[4, 3, 1, 2, 0]> : tensor<5xi32>} : () -> tensor<5xi32>
    %3 = tosa.transpose %1 {perms = array<i32: 4, 3, 1, 2, 0>} : (tensor<42x44x85x24x94xi64>) -> tensor<94x24x44x85x42xi64>
    %4 = tosa.exp %0 : (tensor<f32>) -> tensor<f32>
    %5 = tosa.logical_not %arg3 : (tensor<55x8xi1>) -> tensor<55x8xi1>
    %6 = tosa.reverse %5 {axis = 1 : i32} : (tensor<55x8xi1>) -> tensor<55x8xi1>
    %7 = tosa.abs %5 : (tensor<55x8xi1>) -> tensor<55x8xi1>
    %8 = tosa.maximum %1, %1 : (tensor<42x44x85x24x94xi64>, tensor<42x44x85x24x94xi64>) -> tensor<42x44x85x24x94xi64>
    %9 = tosa.argmax %7 {axis = 0 : i32} : (tensor<55x8xi1>) -> tensor<8xi32>
    %10 = tosa.ceil %4 : (tensor<f32>) -> tensor<f32>
    return %3, %6, %8, %9, %10 : tensor<94x24x44x85x42xi64>, tensor<55x8xi1>, tensor<42x44x85x24x94xi64>, tensor<8xi32>, tensor<f32>
  }
}
