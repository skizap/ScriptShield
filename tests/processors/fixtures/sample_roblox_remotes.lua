-- Simple RemoteEvent
local playerJoinedEvent = game.ReplicatedStorage.PlayerJoined
playerJoinedEvent:FireServer("Player1", 100)

-- RemoteFunction
local getPlayerDataFunc = game.ReplicatedStorage.GetPlayerData
local data = getPlayerDataFunc:InvokeServer(player)

-- Indexed remote access
local remotes = {
    onDamage = game.ReplicatedStorage.OnDamage,
    onHeal = game.ReplicatedStorage.OnHeal
}
remotes.onDamage:FireServer(target, 50)
remotes.onHeal:FireClient(player, 25)

-- Remote in function
function notifyServer(eventName, ...)
    local event = game.ReplicatedStorage:FindFirstChild(eventName)
    if event then
        event:FireServer(...)
    end
end

-- Multiple calls to same remote
local chatEvent = game.ReplicatedStorage.ChatMessage
chatEvent:FireServer("Hello")
chatEvent:FireServer("World")
