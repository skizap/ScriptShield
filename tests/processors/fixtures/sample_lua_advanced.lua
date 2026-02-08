--[[
Comprehensive Lua sample for advanced obfuscation testing.

This module provides a variety of Lua constructs to test advanced
obfuscation features including control flow flattening, dead code injection,
opaque predicates, anti-debugging, code splitting, and self-modifying code.
--]]

-- =============================================================================
-- Basic Functions
-- =============================================================================

-- Simple global function
function simpleAdd(a, b)
    return a + b
end

-- Function with if/else control flow
function checkValue(value)
    if value < 0 then
        return "negative"
    elseif value == 0 then
        return "zero"
    elseif value < 100 then
        return "small positive"
    else
        return "large positive"
    end
end

-- Local function
local function localMultiply(x, y)
    return x * y
end

-- Function with loops
function sumRange(n)
    local total = 0
    for i = 1, n do
        total = total + i
    end
    return total
end

-- While loop function
function countdown(startValue)
    local count = 0
    local current = startValue
    while current > 0 do
        count = count + 1
        current = current - 1
    end
    return count
end

-- Repeat until loop
function repeatExample(limit)
    local i = 0
    repeat
        i = i + 1
    until i >= limit
    return i
end

-- Nested loops
function createMatrix(rows, cols)
    local matrix = {}
    for i = 1, rows do
        matrix[i] = {}
        for j = 1, cols do
            matrix[i][j] = i * j
        end
    end
    return matrix
end

-- =============================================================================
-- Tables and Metatables
-- =============================================================================

-- Simple table
local simpleTable = {
    name = "test",
    value = 42,
    active = true
}

-- Array-style table
local numberArray = {10, 20, 30, 40, 50}

-- Table with methods (class-like)
local Calculator = {}
Calculator.__index = Calculator

function Calculator.new(initialValue)
    local obj = {
        value = initialValue or 0,
        history = {}
    }
    setmetatable(obj, Calculator)
    return obj
end

function Calculator:add(x)
    self.value = self.value + x
    table.insert(self.history, "add " .. tostring(x))
    return self.value
end

function Calculator:subtract(x)
    self.value = self.value - x
    table.insert(self.history, "subtract " .. tostring(x))
    return self.value
end

function Calculator:multiply(x)
    self.value = self.value * x
    table.insert(self.history, "multiply " .. tostring(x))
    return self.value
end

function Calculator:getHistory()
    return self.history
end

-- Data processor class
local DataProcessor = {}
DataProcessor.__index = DataProcessor

function DataProcessor.new(data)
    local obj = {
        data = data or {}
    }
    setmetatable(obj, DataProcessor)
    return obj
end

function DataProcessor:filterPositive()
    local result = {}
    for _, v in ipairs(self.data) do
        if v > 0 then
            table.insert(result, v)
        end
    end
    return result
end

function DataProcessor:mapSquare()
    local result = {}
    for _, v in ipairs(self.data) do
        table.insert(result, v * v)
    end
    return result
end

function DataProcessor:reduceSum()
    local total = 0
    for _, v in ipairs(self.data) do
        total = total + v
    end
    return total
end

-- =============================================================================
-- Nested Functions and Closures
-- =============================================================================

function makeMultiplier(factor)
    return function(x)
        return x * factor
    end
end

function makeCounter()
    local count = 0
    
    local function increment()
        count = count + 1
        return count
    end
    
    local function getCount()
        return count
    end
    
    local function reset()
        count = 0
    end
    
    return {
        increment = increment,
        getCount = getCount,
        reset = reset
    }
end

function outerFunction(x)
    local function middleFunction(y)
        local function innerFunction(z)
            return x + y + z
        end
        return innerFunction
    end
    return middleFunction
end

-- =============================================================================
-- Coroutines (Generators)
-- =============================================================================

function numberGenerator(n)
    return coroutine.create(function()
        for i = 1, n do
            coroutine.yield(i)
        end
    end)
end

function fibonacciGenerator(limit)
    return coroutine.create(function()
        local a, b = 0, 1
        while a < limit do
            coroutine.yield(a)
            a, b = b, a + b
        end
    end)
end

-- =============================================================================
-- Error Handling (pcall/xpcall)
-- =============================================================================

function safeDivision(a, b)
    local success, result = pcall(function()
        return a / b
    end)
    
    if success then
        return result
    else
        return nil, "Division failed"
    end
end

function parseNumber(str)
    local success, value = pcall(function()
        return tonumber(str)
    end)
    
    if success and value then
        return value * 2
    else
        return nil, "Invalid number"
    end
end

-- =============================================================================
-- Functions with Complex Logic
-- =============================================================================

function isPrime(n)
    if n < 2 then
        return false
    end
    if n == 2 then
        return true
    end
    if n % 2 == 0 then
        return false
    end
    
    local sqrt = math.sqrt(n)
    for i = 3, sqrt, 2 do
        if n % i == 0 then
            return false
        end
    end
    return true
end

function factorial(n)
    if n <= 1 then
        return 1
    end
    return n * factorial(n - 1)
end

function binarySearch(arr, target)
    local left = 1
    local right = #arr
    
    while left <= right do
        local mid = math.floor((left + right) / 2)
        
        if arr[mid] == target then
            return mid
        elseif arr[mid] < target then
            left = mid + 1
        else
            right = mid - 1
        end
    end
    
    return -1
end

function quickSort(arr, left, right)
    left = left or 1
    right = right or #arr
    
    if left < right then
        local pivotIndex = partition(arr, left, right)
        quickSort(arr, left, pivotIndex - 1)
        quickSort(arr, pivotIndex + 1, right)
    end
    
    return arr
end

function partition(arr, left, right)
    local pivot = arr[right]
    local i = left - 1
    
    for j = left, right - 1 do
        if arr[j] <= pivot then
            i = i + 1
            arr[i], arr[j] = arr[j], arr[i]
        end
    end
    
    arr[i + 1], arr[right] = arr[right], arr[i + 1]
    return i + 1
end

-- =============================================================================
-- String Processing Functions
-- =============================================================================

function reverseString(str)
    local result = ""
    for i = #str, 1, -1 do
        result = result .. str:sub(i, i)
    end
    return result
end

function countWords(str)
    local count = 0
    for _ in str:gmatch("%S+") do
        count = count + 1
    end
    return count
end

function splitString(str, delimiter)
    local result = {}
    local pattern = "([^" .. delimiter .. "]+)"
    for match in str:gmatch(pattern) do
        table.insert(result, match)
    end
    return result
end

-- =============================================================================
-- Module Pattern
-- =============================================================================

local MyModule = {}

function MyModule.publicFunction(x)
    return x * 2
end

local function privateHelper(y)
    return y + 10
end

function MyModule.process(data)
    return privateHelper(data)
end

-- =============================================================================
-- Main Execution
-- =============================================================================

-- Test basic functions
print(simpleAdd(5, 3))
print(checkValue(50))
print(sumRange(10))

-- Test calculator
local calc = Calculator.new(10)
calc:add(5)
calc:multiply(2)
local history = calc:getHistory()
for _, entry in ipairs(history) do
    print(entry)
end

-- Test closures
local double = makeMultiplier(2)
print(double(7))

-- Test coroutines
local gen = fibonacciGenerator(100)
for num in function() return select(2, coroutine.resume(gen)) end do
    if not coroutine.status(gen) == "suspended" then break end
    print(num)
end

return MyModule
