module {
  func.func @main(%arg0: tensor<70x20x58x86x84xf32>, %arg1: tensor<22x64x4x20x28x78xi8>, %arg2: tensor<1x1x1x20x28x1xi8>, %arg3: tensor<3x5x55x4x40x89xi1>, %arg4: tensor<3x1x1x1x1x1xi1>) -> (tensor<22x64x4x20x28x78xi8>, tensor<3x5x55x4x40x89xi1>, tensor<70x20x58x86x84xf32>) {
    %0 = tosa.sigmoid %arg0 : (tensor<70x20x58x86x84xf32>) -> tensor<70x20x58x86x84xf32>
    %1 = tosa.bitwise_and %arg1, %arg2 : (tensor<22x64x4x20x28x78xi8>, tensor<1x1x1x20x28x1xi8>) -> tensor<22x64x4x20x28x78xi8>
    %2 = tosa.logical_and %arg3, %arg4 : (tensor<3x5x55x4x40x89xi1>, tensor<3x1x1x1x1x1xi1>) -> tensor<3x5x55x4x40x89xi1>
    %3 = tosa.pow %0, %0 : (tensor<70x20x58x86x84xf32>, tensor<70x20x58x86x84xf32>) -> tensor<70x20x58x86x84xf32>
    return %1, %2, %3 : tensor<22x64x4x20x28x78xi8>, tensor<3x5x55x4x40x89xi1>, tensor<70x20x58x86x84xf32>
  }
}
