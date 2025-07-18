module {
  func.func @main(%arg0: tensor<22x96xi16>, %arg1: tensor<81x64x51x68x8xf32>, %arg2: tensor<38x55xi32>, %arg3: tensor<1x1xi32>) -> (tensor<81x64x51x68x8xf32>, tensor<38x55xi32>, tensor<22xi32>, tensor<228x55xi32>, tensor<114x55xi32>) {
    %0 = tosa.reverse %arg0 {axis = 0 : i32} : (tensor<22x96xi16>) -> tensor<22x96xi16>
    %1 = tosa.sigmoid %arg1 : (tensor<81x64x51x68x8xf32>) -> tensor<81x64x51x68x8xf32>
    %2 = tosa.intdiv %arg2, %arg3 : (tensor<38x55xi32>, tensor<1x1xi32>) -> tensor<38x55xi32>
    %3 = "tosa.const"() {values = dense<[0, 1]> : tensor<2xi32>} : () -> tensor<2xi32>
    %4 = tosa.transpose %2 {perms = array<i32: 0, 1>} : (tensor<38x55xi32>) -> tensor<38x55xi32>
    %5 = tosa.argmax %0 {axis = 1 : i32} : (tensor<22x96xi16>) -> tensor<22xi32>
    %t_6 = tosa.const_shape {values = dense<[ 3, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %6 = tosa.tile %2, %t_6 : (tensor<38x55xi32>, !tosa.shape<2>) -> tensor<114x55xi32>
    %7 = tosa.concat %6, %6 {axis = 0 : i32} : (tensor<114x55xi32>, tensor<114x55xi32>) -> tensor<228x55xi32>
    %8 = tosa.intdiv %6, %6 : (tensor<114x55xi32>, tensor<114x55xi32>) -> tensor<114x55xi32>
    return %1, %4, %5, %7, %8 : tensor<81x64x51x68x8xf32>, tensor<38x55xi32>, tensor<22xi32>, tensor<228x55xi32>, tensor<114x55xi32>
  }
}
