module {
  func.func @main(%arg0: tensor<12x68x42x80x47xi1>, %arg1: tensor<54x9x82x13xi64>) -> (tensor<47x80x68x42x12xi1>, tensor<54x9x164x26xi64>) {
    %0 = "tosa.const"() {values = dense<[4, 3, 1, 2, 0]> : tensor<5xi32>} : () -> tensor<5xi32>
    %1 = tosa.transpose %arg0 {perms = array<i32: 4, 3, 1, 2, 0>} : (tensor<12x68x42x80x47xi1>) -> tensor<47x80x68x42x12xi1>
    %t_2 = tosa.const_shape {values = dense<[ 1, 1, 2, 2 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %2 = tosa.tile %arg1, %t_2 : (tensor<54x9x82x13xi64>, !tosa.shape<4>) -> tensor<54x9x164x26xi64>
    return %1, %2 : tensor<47x80x68x42x12xi1>, tensor<54x9x164x26xi64>
  }
}
